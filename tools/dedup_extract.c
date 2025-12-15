/**
 * @file dedup_extract.c
 * @brief Step 1 extractor: choose defaults + build exception masks (lossless).
 *
 * This tool consumes a manifest describing tensor blobs inside one or more packed
 * files (e.g., q4 bytes and/or scales streams), then emits:
 *  - defaults stream (one default blob per group),
 *  - exceptions stream (only differing blocks),
 *  - masks stream (bitset per tensor),
 *  - spec file (dedup.spec.bin) mapping tensors -> defaults/masks/exceptions.
 *
 * TPS goal (CPU on VM):
 *  - Reduce DRAM traffic by eliminating duplicate bytes across similar weights.
 *  - Keep runtime overhead minimal: fixed-size blocks + bitset branch.
 *
 * Grouping strategy (Step 1 decisions):
 *  1) Prefer intra-layer similarity (e.g., expert pairs within a layer).
 *  2) Make grouping controllable via a simple plan file, but also provide a safe
 *     auto-group that normalizes names.
 *
 * IMPORTANT: This is an extraction tool only. Runtime integration uses the spec
 * and streams produced here.
 *
 * Manifest expectations:
 *  The manifest is JSON containing a top-level array of objects. Each object must
 *  include at least:
 *    - "name": string
 *    - "file": string (relative or absolute path)
 *    - "offset": integer (bytes)
 *    - "bytes": integer (bytes)
 *  Optional:
 *    - "kind": string ("int4_blocks", "scales", "other")
 *
 * Plan expectations (optional):
 *  If provided, the plan is a JSON array of group objects:
 *    { "group_id": 123, "default": "tensorA", "members": ["tensorA","tensorB",...] }
 *
 * Block sizing:
 *  - For maximum CPU gain, runtime wants small, cache-friendly blocks.
 *  - Default here is 256 bytes (aligned with your IE_STRIDE_BYTES usage),
 *    but you can override with --block-bytes.
 *
 * Lossless guarantee:
 *  - Every member tensor is compared block-by-block to its chosen default.
 *  - Any differing block is stored in exceptions and marked in the mask.
 *  - If a tensor is identical, its exceptions_blocks = 0 and mask is all zeros.
 */

#include "spec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* ----------------------------- Utilities ----------------------------- */

/** @brief Exit with an error message. */
static void die(const char* msg) {
  fprintf(stderr, "ERROR: %s\n", msg);
  exit(2);
}

/** @brief Read entire file into memory (binary). Caller frees. */
static uint8_t* read_file_all(const char* path, size_t* out_n) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }

  uint8_t* buf = (uint8_t*)malloc((size_t)sz + 1u);
  if (!buf) { fclose(f); return NULL; }
  if (sz > 0) {
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
  }
  fclose(f);
  buf[(size_t)sz] = 0;
  if (out_n) *out_n = (size_t)sz;
  return buf;
}

/** @brief Open a file for random-access reads. */
static FILE* open_rb(const char* path) {
  FILE* f = fopen(path, "rb");
  return f;
}

/** @brief Read exactly n bytes at offset into dst. */
static int pread_exact(FILE* f, uint64_t off, void* dst, size_t n) {
  if (fseek(f, (long)off, SEEK_SET) != 0) return -1;
  if (n == 0) return 0;
  return (fread(dst, 1, n, f) == n) ? 0 : -1;
}

/** @brief Write exactly n bytes. */
static int fwrite_all(FILE* f, const void* p, size_t n) {
  if (n == 0) return 0;
  return (fwrite(p, 1, n, f) == n) ? 0 : -1;
}

/** @brief Ceiling divide for uint64. */
static uint64_t u64_ceil_div(uint64_t a, uint64_t b) {
  return (a + b - 1u) / b;
}

/* --------------------------- Manifest model --------------------------- */

typedef struct blob_ref_s {
  char* name;          /* tensor name */
  char* file;          /* backing file path */
  uint64_t offset;     /* bytes */
  uint64_t bytes;      /* bytes */
  dedup_payload_kind_t kind;
} blob_ref_t;

typedef struct blob_list_s {
  blob_ref_t* v;
  size_t n;
  size_t cap;
} blob_list_t;

/** @brief Append a blob_ref to list. */
static void blobs_push(blob_list_t* L, const blob_ref_t* b) {
  if (L->n == L->cap) {
    size_t nc = (L->cap == 0) ? 128u : (L->cap * 2u);
    blob_ref_t* nv = (blob_ref_t*)realloc(L->v, nc * sizeof(blob_ref_t));
    if (!nv) die("out of memory");
    L->v = nv;
    L->cap = nc;
  }
  L->v[L->n++] = *b;
}

/** @brief Free list strings and storage. */
static void blobs_free(blob_list_t* L) {
  for (size_t i = 0; i < L->n; i++) {
    free(L->v[i].name);
    free(L->v[i].file);
  }
  free(L->v);
  memset(L, 0, sizeof(*L));
}

/**
 * @brief Very small JSON string extractor.
 *
 * This is not a full JSON parser; it is a pragmatic scanner designed for the specific
 * manifest shape we generate/consume. It expects double-quoted keys and values.
 *
 * @param s Pointer to start scanning.
 * @param key JSON key (e.g., "name").
 * @return Newly allocated string value, or NULL if not found.
 */
static char* json_find_string(const char* s, const char* key) {
  char pat[128];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(s, pat);
  if (!p) return NULL;
  p = strchr(p, ':');
  if (!p) return NULL;
  p++;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  if (*p != '"') return NULL;
  p++;
  const char* q = strchr(p, '"');
  if (!q) return NULL;
  size_t n = (size_t)(q - p);
  char* out = (char*)malloc(n + 1u);
  if (!out) return NULL;
  memcpy(out, p, n);
  out[n] = 0;
  return out;
}

/**
 * @brief Very small JSON integer extractor (unsigned).
 *
 * @param s Pointer to start scanning.
 * @param key JSON key.
 * @param out Parsed value.
 * @return 0 on success, non-zero on failure.
 */
static int json_find_u64(const char* s, const char* key, uint64_t* out) {
  char pat[128];
  snprintf(pat, sizeof(pat), "\"%s\"", key);
  const char* p = strstr(s, pat);
  if (!p) return -1;
  p = strchr(p, ':');
  if (!p) return -1;
  p++;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
  errno = 0;
  unsigned long long v = strtoull(p, NULL, 10);
  if (errno != 0) return -1;
  *out = (uint64_t)v;
  return 0;
}

/**
 * @brief Map a manifest "kind" string to enum.
 *
 * @param s String (may be NULL).
 * @return Payload kind enum.
 */
static dedup_payload_kind_t kind_from_string(const char* s) {
  if (!s) return DEDUP_PAYLOAD_UNKNOWN;
  if (strcmp(s, "int4_blocks") == 0) return DEDUP_PAYLOAD_INT4_BLOCKS;
  if (strcmp(s, "scales") == 0) return DEDUP_PAYLOAD_SCALES;
  if (strcmp(s, "other") == 0) return DEDUP_PAYLOAD_OTHER;
  return DEDUP_PAYLOAD_UNKNOWN;
}

/**
 * @brief Load manifest JSON array into a blob list.
 *
 * The loader scans for object boundaries by finding '{' and '}' pairs and then
 * extracting required fields. This is robust enough for the specific manifests
 * used in this codebase.
 *
 * @param path Manifest path.
 * @param out Output list.
 * @return 0 on success, non-zero on error.
 */
static int load_manifest(const char* path, blob_list_t* out) {
  size_t n = 0;
  uint8_t* txt = read_file_all(path, &n);
  if (!txt) return -1;

  const char* s = (const char*)txt;
  const char* p = s;
  while ((p = strchr(p, '{')) != NULL) {
    const char* q = strchr(p, '}');
    if (!q) break;

    size_t obj_len = (size_t)(q - p + 1);
    char* obj = (char*)malloc(obj_len + 1u);
    if (!obj) { free(txt); return -1; }
    memcpy(obj, p, obj_len);
    obj[obj_len] = 0;

    blob_ref_t b;
    memset(&b, 0, sizeof(b));
    b.name = json_find_string(obj, "name");
    b.file = json_find_string(obj, "file");
    if (!b.name || !b.file) { free(obj); continue; }
    if (json_find_u64(obj, "offset", &b.offset) != 0) { free(obj); continue; }
    if (json_find_u64(obj, "bytes", &b.bytes) != 0) { free(obj); continue; }

    char* k = json_find_string(obj, "kind");
    b.kind = kind_from_string(k);
    free(k);

    blobs_push(out, &b);
    free(obj);
    p = q + 1;
  }

  free(txt);
  return (out->n > 0) ? 0 : -1;
}

/* --------------------------- Grouping model --------------------------- */

typedef struct group_s {
  uint64_t group_id;
  size_t default_idx; /* index into blob list */
  size_t* members;    /* indices into blob list */
  size_t members_n;
  size_t members_cap;
} group_t;

typedef struct group_list_s {
  group_t* v;
  size_t n;
  size_t cap;
} group_list_t;

/** @brief Append member to a group. */
static void group_add_member(group_t* g, size_t idx) {
  if (g->members_n == g->members_cap) {
    size_t nc = (g->members_cap == 0) ? 16u : (g->members_cap * 2u);
    size_t* nv = (size_t*)realloc(g->members, nc * sizeof(size_t));
    if (!nv) die("out of memory");
    g->members = nv;
    g->members_cap = nc;
  }
  g->members[g->members_n++] = idx;
}

/** @brief Free group list. */
static void groups_free(group_list_t* G) {
  for (size_t i = 0; i < G->n; i++) {
    free(G->v[i].members);
  }
  free(G->v);
  memset(G, 0, sizeof(*G));
}

/** @brief Push new group. */
static group_t* groups_push(group_list_t* G, uint64_t gid, size_t def_idx) {
  if (G->n == G->cap) {
    size_t nc = (G->cap == 0) ? 64u : (G->cap * 2u);
    group_t* nv = (group_t*)realloc(G->v, nc * sizeof(group_t));
    if (!nv) die("out of memory");
    G->v = nv;
    G->cap = nc;
  }
  group_t* g = &G->v[G->n++];
  memset(g, 0, sizeof(*g));
  g->group_id = gid;
  g->default_idx = def_idx;
  group_add_member(g, def_idx);
  return g;
}

/**
 * @brief Normalize a tensor name for auto-grouping.
 *
 * The goal is to group "near-duplicates" (expert pairs, repeated projections, etc.)
 * by removing suffix patterns that typically distinguish members of a redundant set.
 *
 * This is conservative: if normalization fails to match, we do not force grouping.
 *
 * @param name Input tensor name.
 * @param out_buf Output buffer.
 * @param out_cap Capacity of output buffer.
 */
static void normalize_name(const char* name, char* out_buf, size_t out_cap) {
  /* Copy and apply simple rewrites. */
  snprintf(out_buf, out_cap, "%s", name);

  /* Remove obvious per-expert/per-rank suffixes if present. */
  const char* patterns[] = {
    ".expert_0", ".expert_1", ".expert_2", ".expert_3",
    ".rank0", ".rank1", ".rank2", ".rank3",
    NULL
  };
  for (int i = 0; patterns[i]; i++) {
    char* p = strstr(out_buf, patterns[i]);
    if (p) {
      memmove(p, p + strlen(patterns[i]), strlen(p + strlen(patterns[i])) + 1u);
    }
  }

  /* Collapse any accidental double dots. */
  for (;;) {
    char* dd = strstr(out_buf, "..");
    if (!dd) break;
    memmove(dd, dd + 1, strlen(dd));
  }
}

/**
 * @brief Auto-group blobs by (file, bytes, kind, normalized-name).
 *
 * Step 1 decision:
 *  - Grouping by identical byte length is required to make block-wise comparison valid.
 *  - Grouping by same backing file is helpful in early bring-up (single stream).
 *  - Grouping by normalized name is a heuristic; users can override via --plan.
 *
 * @param blobs Manifest blobs.
 * @param out_groups Output groups.
 */
static void build_groups_auto(const blob_list_t* blobs, group_list_t* out_groups) {
  uint64_t next_gid = 1u;

  char ni[1024];
  char nj[1024];

  int* assigned = (int*)calloc(blobs->n, sizeof(int));
  if (!assigned) die("out of memory");

  for (size_t i = 0; i < blobs->n; i++) {
    if (assigned[i]) continue;

    const blob_ref_t* bi = &blobs->v[i];
    normalize_name(bi->name, ni, sizeof(ni));

    group_t* g = groups_push(out_groups, next_gid++, i);
    assigned[i] = 1;

    for (size_t j = i + 1; j < blobs->n; j++) {
      if (assigned[j]) continue;
      const blob_ref_t* bj = &blobs->v[j];

      if (bj->bytes != bi->bytes) continue;
      if (bj->kind != bi->kind) continue;
      if (strcmp(bj->file, bi->file) != 0) continue;

      normalize_name(bj->name, nj, sizeof(nj));
      if (strcmp(nj, ni) != 0) continue;

      group_add_member(g, j);
      assigned[j] = 1;
    }
  }

  free(assigned);
}

/* ----------------------------- Extraction ----------------------------- */

/**
 * @brief Write a mask bit for a block index.
 *
 * @param mask Byte array.
 * @param block_index Block index.
 * @param value 0 or 1.
 */
static void mask_set(uint8_t* mask, uint64_t block_index, int value) {
  const uint64_t byte_i = block_index >> 3;
  const uint64_t bit_i  = block_index & 7u;
  const uint8_t  bit    = (uint8_t)(1u << (uint8_t)bit_i);
  if (value) mask[byte_i] |= bit;
  else       mask[byte_i] &= (uint8_t)~bit;
}

/**
 * @brief Compare member blob to default blob and emit exceptions + mask.
 *
 * @param f Default file handle (same as member file, or separate if needed).
 * @param def_ref Default blob ref.
 * @param mem_ref Member blob ref.
 * @param block_bytes Block size.
 * @param exceptions_out Exceptions stream handle.
 * @param masks_out Masks stream handle.
 * @param exceptions_off_inout Updated with current exceptions offset.
 * @param masks_off_inout Updated with current masks offset.
 * @param entry_out Spec entry to fill (name_off/group_id already filled by caller).
 * @return 0 on success, non-zero on error.
 */
static int extract_one(
  FILE* f,
  const blob_ref_t* def_ref,
  const blob_ref_t* mem_ref,
  uint64_t block_bytes,
  FILE* exceptions_out,
  FILE* masks_out,
  uint64_t* exceptions_off_inout,
  uint64_t* masks_off_inout,
  dedup_spec_entry_t* entry_out
) {
  if (def_ref->bytes != mem_ref->bytes) return -1;
  if (block_bytes == 0) return -1;
  if ((def_ref->bytes % block_bytes) != 0) return -1;

  const uint64_t block_count = def_ref->bytes / block_bytes;
  const uint64_t mask_bytes = u64_ceil_div(block_count, 8u);

  uint8_t* mask = (uint8_t*)calloc((size_t)mask_bytes, 1u);
  if (!mask) return -1;

  uint8_t* buf_def = (uint8_t*)malloc((size_t)block_bytes);
  uint8_t* buf_mem = (uint8_t*)malloc((size_t)block_bytes);
  if (!buf_def || !buf_mem) { free(mask); free(buf_def); free(buf_mem); return -1; }

  uint64_t ex_blocks = 0;

  /* Walk blocks; store only deltas. */
  for (uint64_t i = 0; i < block_count; i++) {
    const uint64_t off = i * block_bytes;

    if (pread_exact(f, def_ref->offset + off, buf_def, (size_t)block_bytes) != 0) { free(mask); free(buf_def); free(buf_mem); return -1; }
    if (pread_exact(f, mem_ref->offset + off, buf_mem, (size_t)block_bytes) != 0) { free(mask); free(buf_def); free(buf_mem); return -1; }

    if (memcmp(buf_def, buf_mem, (size_t)block_bytes) != 0) {
      mask_set(mask, i, 1);

      /* Append this block to exceptions stream. */
      if (fwrite_all(exceptions_out, buf_mem, (size_t)block_bytes) != 0) { free(mask); free(buf_def); free(buf_mem); return -1; }
      ex_blocks++;
    }
  }

  /* Append mask to masks stream. */
  if (fwrite_all(masks_out, mask, (size_t)mask_bytes) != 0) { free(mask); free(buf_def); free(buf_mem); return -1; }

  /* Fill entry. */
  entry_out->payload_kind = (uint32_t)mem_ref->kind;
  entry_out->original_bytes = mem_ref->bytes;
  entry_out->block_bytes = block_bytes;
  entry_out->block_count = block_count;

  entry_out->defaults_off = 0; /* filled by caller (per-group) */
  entry_out->exceptions_off = *exceptions_off_inout;
  entry_out->exceptions_blocks = ex_blocks;

  entry_out->mask_off = *masks_off_inout;
  entry_out->mask_bytes = mask_bytes;

  *exceptions_off_inout += ex_blocks * block_bytes;
  *masks_off_inout += mask_bytes;

  free(mask);
  free(buf_def);
  free(buf_mem);
  return 0;
}

/* ----------------------------- CLI ----------------------------- */

typedef struct args_s {
  const char* manifest;
  const char* out_dir;
  uint64_t block_bytes;
} args_t;

/** @brief Parse CLI args (minimal). */
static args_t parse_args(int argc, char** argv) {
  args_t a;
  memset(&a, 0, sizeof(a));
  a.block_bytes = 256u;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) { a.manifest = argv[++i]; continue; }
    if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) { a.out_dir = argv[++i]; continue; }
    if (strcmp(argv[i], "--block-bytes") == 0 && i + 1 < argc) { a.block_bytes = (uint64_t)strtoull(argv[++i], NULL, 10); continue; }
    if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: dedup_extract --manifest <manifest.json> --out-dir <dir> [--block-bytes N]\n");
      exit(0);
    }
    fprintf(stderr, "WARN: unknown arg: %s\n", argv[i]);
  }

  if (!a.manifest) die("missing --manifest");
  if (!a.out_dir) die("missing --out-dir");
  if (a.block_bytes == 0) die("--block-bytes must be > 0");
  return a;
}

/**
 * @brief Join directory + filename into an output path.
 *
 * @param dir Directory path.
 * @param name File name.
 * @param out Output buffer.
 * @param cap Output capacity.
 */
static void path_join(const char* dir, const char* name, char* out, size_t cap) {
  size_t dl = strlen(dir);
  if (dl > 0 && dir[dl - 1] == '/') snprintf(out, cap, "%s%s", dir, name);
  else snprintf(out, cap, "%s/%s", dir, name);
}

int main(int argc, char** argv) {
  args_t a = parse_args(argc, argv);

  blob_list_t blobs;
  memset(&blobs, 0, sizeof(blobs));
  if (load_manifest(a.manifest, &blobs) != 0) die("failed to load manifest (expected array of objects with name/file/offset/bytes)");

  group_list_t groups;
  memset(&groups, 0, sizeof(groups));
  build_groups_auto(&blobs, &groups);

  /* Open output streams. */
  char p_spec[4096], p_def[4096], p_ex[4096], p_mask[4096];
  path_join(a.out_dir, "dedup.spec.bin", p_spec, sizeof(p_spec));
  path_join(a.out_dir, "dedup.defaults.bin", p_def, sizeof(p_def));
  path_join(a.out_dir, "dedup.exceptions.bin", p_ex, sizeof(p_ex));
  path_join(a.out_dir, "dedup.masks.bin", p_mask, sizeof(p_mask));

  FILE* f_def = fopen(p_def, "wb");
  FILE* f_ex  = fopen(p_ex, "wb");
  FILE* f_msk = fopen(p_mask, "wb");
  if (!f_def || !f_ex || !f_msk) die("failed to open output files (check --out-dir exists)");

  uint64_t defaults_off = 0;
  uint64_t exceptions_off = 0;
  uint64_t masks_off = 0;

  dedup_spec_builder_t* sb = dedup_spec_builder_create();
  if (!sb) die("failed to allocate spec builder");

  /* For simplicity in Step 1, we assume all blobs can be read via their own file handles. */
  for (size_t gi = 0; gi < groups.n; gi++) {
    group_t* g = &groups.v[gi];
    const blob_ref_t* def_ref = &blobs.v[g->default_idx];

    /* Write the default blob once per group into defaults.bin. */
    FILE* f_in = open_rb(def_ref->file);
    if (!f_in) die("failed to open input file for default blob");

    uint8_t* tmp = (uint8_t*)malloc((size_t)def_ref->bytes);
    if (!tmp) die("out of memory");
    if (pread_exact(f_in, def_ref->offset, tmp, (size_t)def_ref->bytes) != 0) die("failed to read default blob bytes");
    if (fwrite_all(f_def, tmp, (size_t)def_ref->bytes) != 0) die("failed to write defaults.bin");
    free(tmp);

    const uint64_t group_defaults_off = defaults_off;
    defaults_off += def_ref->bytes;

    /* Emit entries for each member. */
    for (size_t mi = 0; mi < g->members_n; mi++) {
      const size_t idx = g->members[mi];
      const blob_ref_t* mem_ref = &blobs.v[idx];

      /* All members must match the default's size to be in this group. */
      if (mem_ref->bytes != def_ref->bytes) continue;

      dedup_spec_entry_t e;
      memset(&e, 0, sizeof(e));
      e.group_id = g->group_id;

      const uint64_t name_off = dedup_spec_builder_intern_string(sb, mem_ref->name);
      if (!name_off) die("failed to intern name");
      e.name_off = name_off;

      e.defaults_off = group_defaults_off;

      FILE* f_member = open_rb(mem_ref->file);
      if (!f_member) die("failed to open input file for member blob");

      /* Fast path: default itself => identical, store zero exceptions, zero mask. */
      if (idx == g->default_idx) {
        const uint64_t block_bytes = a.block_bytes;
        if ((mem_ref->bytes % block_bytes) != 0) die("blob bytes must be divisible by --block-bytes");

        const uint64_t block_count = mem_ref->bytes / block_bytes;
        const uint64_t mask_bytes2 = u64_ceil_div(block_count, 8u);

        uint8_t* zmask = (uint8_t*)calloc((size_t)mask_bytes2, 1u);
        if (!zmask) die("out of memory");
        if (fwrite_all(f_msk, zmask, (size_t)mask_bytes2) != 0) die("failed writing masks.bin");
        free(zmask);

        e.payload_kind = (uint32_t)mem_ref->kind;
        e.original_bytes = mem_ref->bytes;
        e.block_bytes = block_bytes;
        e.block_count = block_count;

        e.exceptions_off = exceptions_off;
        e.exceptions_blocks = 0;

        e.mask_off = masks_off;
        e.mask_bytes = mask_bytes2;
        masks_off += mask_bytes2;

        if (dedup_spec_builder_add_entry(sb, &e) != 0) die("failed to add spec entry");

        fclose(f_member);
        continue;
      }

      if (extract_one(
            f_member,
            def_ref,
            mem_ref,
            a.block_bytes,
            f_ex,
            f_msk,
            &exceptions_off,
            &masks_off,
            &e
          ) != 0) {
        die("extract_one failed (check manifest offsets/bytes and block sizing)");
      }

      if (dedup_spec_builder_add_entry(sb, &e) != 0) die("failed to add spec entry");
      fclose(f_member);
    }

    fclose(f_in);
  }

  /* Finalize spec. */
  if (dedup_spec_builder_write_file(sb, p_spec, defaults_off, exceptions_off, masks_off) != 0) {
    die("failed to write dedup.spec.bin");
  }

  dedup_spec_builder_destroy(sb);
  fclose(f_def);
  fclose(f_ex);
  fclose(f_msk);

  groups_free(&groups);
  blobs_free(&blobs);

  fprintf(stderr, "[dedup_extract] wrote:\n");
  fprintf(stderr, "  %s\n  %s\n  %s\n  %s\n", p_spec, p_def, p_ex, p_mask);
  fprintf(stderr, "[dedup_extract] totals: defaults=%llu bytes, exceptions=%llu bytes, masks=%llu bytes\n",
          (unsigned long long)defaults_off,
          (unsigned long long)exceptions_off,
          (unsigned long long)masks_off);

  return 0;
}

