/**
 * @file ie_kernels_ze.cpp
 * @brief Intel oneAPI Level Zero backend: GEMV/activation/packing launchers.
 *
 * This TU owns the Level Zero runtime interaction (context, device, queue,
 * module, kernels) and exposes a **C ABI** so the C11 core can call it without
 * pulling Level Zero headers elsewhere.
 *
 * ### What this provides
 * - Row-wise GEMV (FP32): y = alpha * W * x + beta * y
 * - Fused GEMV + bias + activation (ReLU/Tanh)
 * - Vector Tanh (FP32)
 * - Weight packing (Blocked-K)
 *
 * ### How kernels are provided
 * Kernels are written in OpenCL C (see `ie_kernels_ze.cl`) and compiled to
 * SPIR-V. At runtime, this TU loads a single SPIR-V module file that contains
 * all kernels. The default search path is:
 *   - $IE_ZE_SPV (env var; if set)
 *   - engine/src/kernels/ze/ie_kernels_ze.spv (repo path)
 *
 * ### Pointer semantics
 * The public C API here accepts **host pointers**. The launcher allocates
 * device buffers, copies inputs to the GPU, dispatches the kernel, and copies
 * the outputs back. That keeps the C11 core simple and portable.
 *
 * ### Error model
 * All public functions return 0 on success or a **negative** value on error.
 * Use ::ie_ze_last_error_string() for a human-readable diagnostic.
 *
 * @ingroup IE_GPU_ZE
 */

#include <level_zero/ze_api.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

/* =============================== C ABI ==================================== */
extern "C" {

/** @defgroup IE_GPU_ZE Level Zero (oneAPI) GPU backend
 *  @brief C-callable API to launch kernels via Intel Level Zero.
 *  @{ */

/**
 * @brief Activation kinds (must match the kernel’s expectations).
 */
typedef enum ie_act_kind_e {
  IE_ACT_NONE = 0,  /**< Identity.   */
  IE_ACT_RELU = 1,  /**< ReLU.       */
  IE_ACT_TANH = 2   /**< tanh().     */
} ie_act_kind_t;

/**
 * @brief Get the last error message set by this backend (thread-local).
 * @return NUL-terminated message; never NULL (may be empty string).
 */
const char* ie_ze_last_error_string(void);

/**
 * @brief Row-wise GEMV FP32: y = alpha * W * x + beta * y.
 *
 * @param W     Host pointer to row-major matrix (rows x ldW).
 * @param x     Host pointer to vector (length cols).
 * @param y     Host pointer to output vector (length rows).
 * @param rows  Number of rows in W and elements in y (>=1).
 * @param cols  Number of columns in W and elements in x (>=1).
 * @param ldW   Leading dimension of W (>= cols).
 * @param alpha Scale for W*x.
 * @param beta  Scale for existing y (0 to overwrite).
 * @return 0 on success; negative on error (see ::ie_ze_last_error_string()).
 */
int ie_ze_launch_gemv_rowwise_f32(const float* W,
                                  const float* x,
                                  float*       y,
                                  int          rows,
                                  int          cols,
                                  int          ldW,
                                  float        alpha,
                                  float        beta);

/**
 * @brief Fused GEMV + bias + activation: y = act(alpha*W*x + bias + beta*y).
 *
 * @param W     Host pointer to row-major matrix (rows x ldW).
 * @param x     Host pointer to vector (length cols).
 * @param bias  Host pointer to per-row bias (length rows) or NULL.
 * @param y     Host pointer to output vector (length rows).
 * @param rows  Number of rows.
 * @param cols  Number of cols.
 * @param ldW   Leading dimension of W (>= cols).
 * @param alpha Scale for W*x.
 * @param beta  Scale for existing y (0 to overwrite).
 * @param act   Activation kind (ie_act_kind_t).
 * @return 0 on success; negative on error.
 */
int ie_ze_launch_gemv_bias_act_f32(const float* W,
                                   const float* x,
                                   const float* bias,
                                   float*       y,
                                   int          rows,
                                   int          cols,
                                   int          ldW,
                                   float        alpha,
                                   float        beta,
                                   ie_act_kind_t act);

/**
 * @brief Vector tanh: y[i] = tanh(x[i]) for i in [0, n).
 *
 * @param y    Host pointer to output (length n).
 * @param x    Host pointer to input  (length n).
 * @param n    Number of elements (>=1).
 * @return 0 on success; negative on error.
 */
int ie_ze_launch_vec_tanh_f32(float*       y,
                              const float* x,
                              int          n);

/**
 * @brief Pack W (row-major) into Blocked-K layout into Wp.
 *
 * @param Wp      Host pointer to destination (size rows*cols).
 * @param W       Host pointer to source row-major matrix.
 * @param rows    Rows (>=1).
 * @param cols    Cols (>=1).
 * @param ldW     Leading dimension of W (>= cols).
 * @param block_k Tile size along K (e.g., 32/64/128; >0).
 * @return 0 on success; negative on error.
 */
int ie_ze_launch_pack_w_blockedk_f32(float*       Wp,
                                     const float* W,
                                     int          rows,
                                     int          cols,
                                     int          ldW,
                                     int          block_k);

/** @} */ /* end group IE_GPU_ZE */
} /* extern "C" */

/* ============================ internal helpers ============================ */

/** Thread-local error buffer. */
static thread_local char g_err[512] = {0};

/**
 * @brief Set last error (thread-local); truncates if needed.
 * @param s Message or NULL to clear.
 */
static void set_err(const char* s) {
  if (!s) { g_err[0] = '\0'; return; }
  std::snprintf(g_err, sizeof(g_err), "%s", s);
}

/** ZE guard macro: store message and return negative code on failure. */
#define ZE_GUARD(expr, msg)                                         \
  do {                                                              \
    ze_result_t _st = (expr);                                       \
    if (_st != ZE_RESULT_SUCCESS) {                                 \
      std::ostringstream _oss;                                      \
      _oss << (msg) << " (ze_result=" << (int)_st << ")";           \
      set_err(_oss.str().c_str());                                  \
      return -(int)_st;                                             \
    }                                                               \
  } while (0)

/* -------- persistent ZE objects (created on first use, reused) ----------- */

static ze_context_handle_t        g_ctx   = nullptr;
static ze_device_handle_t         g_dev   = nullptr;
static ze_command_queue_handle_t  g_q     = nullptr;
static ze_command_list_handle_t   g_cl    = nullptr; /* immediate list */

static ze_module_handle_t         g_mod   = nullptr;
static ze_kernel_handle_t         g_k_gemv      = nullptr;
static ze_kernel_handle_t         g_k_gemv_ba   = nullptr;
static ze_kernel_handle_t         g_k_tanh      = nullptr;
static ze_kernel_handle_t         g_k_pack      = nullptr;

/* -------------------------- utility: file IO ----------------------------- */

/**
 * @brief Read a whole binary file into memory.
 * @param path filesystem path.
 * @param out  vector to fill.
 * @return true on success.
 */
static bool read_file(const std::string& path, std::vector<uint8_t>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  f.seekg(0, std::ios::end);
  std::streampos sz = f.tellg();
  if (sz <= 0) return false;
  out.resize((size_t)sz);
  f.seekg(0, std::ios::beg);
  f.read(reinterpret_cast<char*>(out.data()), sz);
  return f.good();
}

/* --------------------- init: context/device/queue ------------------------ */

/**
 * @brief Ensure Level Zero context, device and immediate command list exist.
 *        Picks the first GPU device on the first driver found.
 * @return 0 on success; negative on error.
 */
static int ensure_ze_ready() {
  if (g_ctx && g_dev && g_q && g_cl) return 0;

  ZE_GUARD(zeInit(0), "zeInit failed");

  uint32_t nd = 0;
  ZE_GUARD(zeDriverGet(&nd, nullptr), "zeDriverGet(count) failed");
  if (nd == 0) { set_err("no Level Zero drivers found"); return -1; }
  std::vector<ze_driver_handle_t> drivers(nd);
  ZE_GUARD(zeDriverGet(&nd, drivers.data()), "zeDriverGet(list) failed");

  ze_driver_handle_t drv = drivers[0];

  uint32_t ndev = 0;
  ZE_GUARD(zeDeviceGet(drv, &ndev, nullptr), "zeDeviceGet(count) failed");
  if (ndev == 0) { set_err("no Level Zero devices found"); return -1; }
  std::vector<ze_device_handle_t> devs(ndev);
  ZE_GUARD(zeDeviceGet(drv, &ndev, devs.data()), "zeDeviceGet(list) failed");

  /* pick first GPU if possible */
  for (auto d : devs) {
    ze_device_properties_t p = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    zeDeviceGetProperties(d, &p);
    if (p.type == ZE_DEVICE_TYPE_GPU) { g_dev = d; break; }
  }
  if (!g_dev) g_dev = devs[0]; /* fallback */

  ze_context_desc_t cdesc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
  ZE_GUARD(zeContextCreate(drv, &cdesc, &g_ctx), "zeContextCreate failed");

  /* find a compute queue group ordinal */
  uint32_t ng = 0;
  zeDeviceGetCommandQueueGroupProperties(g_dev, &ng, nullptr);
  std::vector<ze_command_queue_group_properties_t> gps(
      ng, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
  zeDeviceGetCommandQueueGroupProperties(g_dev, &ng, gps.data());
  uint32_t ord = 0;
  for (uint32_t i = 0; i < ng; ++i) {
    if (gps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      ord = i; break;
    }
  }

  ze_command_queue_desc_t qd = {
    ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, /* stype            */
    nullptr,                               /* pNext            */
    ord,                                   /* ordinal          */
    0,                                     /* index            */
    ZE_COMMAND_QUEUE_MODE_DEFAULT,
    ZE_COMMAND_QUEUE_PRIORITY_NORMAL
  };
  ZE_GUARD(zeCommandQueueCreate(g_ctx, g_dev, &qd, &g_q), "zeCommandQueueCreate failed");

  /* Immediate command list to reduce submission overhead. */
  ze_command_list_desc_t ld = {
    ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, ord, 0
  };
  ZE_GUARD(zeCommandListCreateImmediate(g_ctx, g_dev, &ld, &g_cl),
           "zeCommandListCreateImmediate failed");

  return 0;
}

/* -------------------- module & kernels: lazy load ------------------------ */

static const char* kDefaultSPVPath = "engine/src/kernels/ze/ie_kernels_ze.spv";

/**
 * @brief Ensure the SPIR-V module and all kernel handles are created.
 * @return 0 on success; negative on error.
 */
static int ensure_module_ready() {
  if (g_mod && g_k_gemv && g_k_gemv_ba && g_k_tanh && g_k_pack) return 0;

  int rc = ensure_ze_ready();
  if (rc < 0) return rc;

  std::string path;
  if (const char* env = std::getenv("IE_ZE_SPV")) path = env;
  if (path.empty()) path = kDefaultSPVPath;

  std::vector<uint8_t> bin;
  if (!read_file(path, bin)) {
    set_err(("failed to read SPIR-V: " + path).c_str());
    return -1;
  }

  ze_module_desc_t md = {};
  md.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
  md.format       = ZE_MODULE_FORMAT_IL_SPIRV;
  md.pInputModule = bin.data();
  md.inputSize    = bin.size();
  md.pBuildFlags  = "";

  ze_module_build_log_handle_t blog = nullptr;
  ze_result_t mrc = zeModuleCreate(g_ctx, g_dev, &md, &g_mod, &blog);

  /* If there is a build log, fetch it on error for diagnostics. */
  if (mrc != ZE_RESULT_SUCCESS) {
    size_t lsz = 0;
    if (blog) zeModuleBuildLogGetString(blog, &lsz, nullptr);
    std::string log;
    if (blog && lsz > 1) {
      log.resize(lsz);
      zeModuleBuildLogGetString(blog, &lsz, log.data());
    }
    if (blog) zeModuleBuildLogDestroy(blog);
    std::ostringstream oss; oss << "zeModuleCreate failed; log: " << log;
    set_err(oss.str().c_str());
    return -(int)mrc;
  }
  if (blog) zeModuleBuildLogDestroy(blog);

  auto mk = [&](const char* name, ze_kernel_handle_t& out,
                uint32_t gx, uint32_t gy, uint32_t gz) -> int {
    ze_kernel_desc_t kd = {ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, 0, name};
    ZE_GUARD(zeKernelCreate(g_mod, &kd, &out), "zeKernelCreate failed");
    /* Use fixed group size for predictability (tuned to 256 threads total). */
    ZE_GUARD(zeKernelSetGroupSize(out, gx, gy, gz), "zeKernelSetGroupSize failed");
    return 0;
  };

  /* Names must match ie_kernels_ze.cl */
  ZE_GUARD(mk("gemv_rowwise_f32", g_k_gemv,    256, 1, 1), "mk(gemv_rowwise_f32) failed");
  ZE_GUARD(mk("gemv_bias_act_f32", g_k_gemv_ba, 256, 1, 1), "mk(gemv_bias_act_f32) failed");
  ZE_GUARD(mk("vec_tanh_f32",      g_k_tanh,    256, 1, 1), "mk(vec_tanh_f32) failed");
  ZE_GUARD(mk("pack_w_blockedk_f32", g_k_pack,   32,  8, 1), "mk(pack_w_blockedk_f32) failed");

  return 0;
}

/* ------------------------ memory helpers (USM) --------------------------- */

/**
 * @brief Allocate device memory.
 * @param bytes Size.
 * @param ptr   Out device pointer.
 * @return 0 on success; negative on error.
 */
static int dalloc(size_t bytes, void** ptr) {
  ze_device_mem_alloc_desc_t ad = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
  ad.flags = 0; ad.ordinal = 0;
  ZE_GUARD(zeMemAllocDevice(g_ctx, &ad, bytes, /*alignment*/64, g_dev, ptr),
           "zeMemAllocDevice failed");
  return 0;
}

/**
 * @brief Free USM pointer if non-null.
 * @param p pointer.
 */
static void dfree(void* p) {
  if (p) zeMemFree(g_ctx, p);
}

/**
 * @brief Copy host→device or device→host using immediate command list.
 * @param dst destination pointer.
 * @param src source pointer.
 * @param bytes size in bytes.
 * @return 0 on success; negative on error.
 */
static int memcpy_imm(void* dst, const void* src, size_t bytes) {
  ZE_GUARD(zeCommandListAppendMemoryCopy(g_cl, dst, src, bytes, nullptr, 0, nullptr),
           "zeCommandListAppendMemoryCopy failed");
  /* Immediate list executes synchronously; no extra sync required. */
  return 0;
}

/* ---------------------------- arg setting -------------------------------- */

/**
 * @brief Helper to set a POD kernel argument.
 */
template <typename T>
static inline ze_result_t set_arg(ze_kernel_handle_t k, uint32_t idx, const T& v) {
  return zeKernelSetArgumentValue(k, idx, sizeof(T), &v);
}

/* ============================== C API impl =============================== */

extern "C" {

/* ------------------------------ errors ----------------------------------- */

const char* ie_ze_last_error_string(void) {
  return g_err;
}

/* ------------------------------ GEMV ------------------------------------- */

/**
 * @copydoc ie_ze_launch_gemv_rowwise_f32
 */
int ie_ze_launch_gemv_rowwise_f32(const float* W,
                                  const float* x,
                                  float*       y,
                                  int          rows,
                                  int          cols,
                                  int          ldW,
                                  float        alpha,
                                  float        beta) {
  set_err(nullptr);
  if (!W || !x || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    set_err("invalid arguments");
    return -1;
  }
  int rc = ensure_module_ready();
  if (rc < 0) return rc;

  size_t szW = (size_t)rows * (size_t)ldW * sizeof(float);
  size_t szx = (size_t)cols * sizeof(float);
  size_t szy = (size_t)rows * sizeof(float);

  void *dW=nullptr, *dx=nullptr, *dy=nullptr;
  if ((rc = dalloc(szW, &dW)) < 0) return rc;
  if ((rc = dalloc(szx, &dx)) < 0) { dfree(dW); return rc; }
  if ((rc = dalloc(szy, &dy)) < 0) { dfree(dx); dfree(dW); return rc; }

  if ((rc = memcpy_imm(dW, W, szW)) < 0) goto CLEANUP;
  if ((rc = memcpy_imm(dx, x, szx)) < 0) goto CLEANUP;
  if (beta != 0.f) { if ((rc = memcpy_imm(dy, y, szy)) < 0) goto CLEANUP; }

  /* Set arguments: (W, x, y, rows, cols, ldW, alpha, beta) */
  ZE_GUARD(set_arg(g_k_gemv, 0, dW),   "set_arg W failed");
  ZE_GUARD(set_arg(g_k_gemv, 1, dx),   "set_arg x failed");
  ZE_GUARD(set_arg(g_k_gemv, 2, dy),   "set_arg y failed");
  ZE_GUARD(set_arg(g_k_gemv, 3, rows), "set_arg rows failed");
  ZE_GUARD(set_arg(g_k_gemv, 4, cols), "set_arg cols failed");
  ZE_GUARD(set_arg(g_k_gemv, 5, ldW),  "set_arg ldW failed");
  ZE_GUARD(set_arg(g_k_gemv, 6, alpha),"set_arg alpha failed");
  ZE_GUARD(set_arg(g_k_gemv, 7, beta), "set_arg beta failed");

  ze_group_count_t gc = { (uint32_t)rows, 1u, 1u };
  ZE_GUARD(zeCommandListAppendLaunchKernel(g_cl, g_k_gemv, &gc, nullptr, 0, nullptr),
           "launch gemv_rowwise_f32 failed");

  if ((rc = memcpy_imm(y, dy, szy)) < 0) goto CLEANUP;

CLEANUP:
  dfree(dy); dfree(dx); dfree(dW);
  return rc;
}

/**
 * @copydoc ie_ze_launch_gemv_bias_act_f32
 */
int ie_ze_launch_gemv_bias_act_f32(const float* W,
                                   const float* x,
                                   const float* bias,
                                   float*       y,
                                   int          rows,
                                   int          cols,
                                   int          ldW,
                                   float        alpha,
                                   float        beta,
                                   ie_act_kind_t act) {
  set_err(nullptr);
  if (!W || !x || !y || rows <= 0 || cols <= 0 || ldW < cols) {
    set_err("invalid arguments");
    return -1;
  }
  int rc = ensure_module_ready();
  if (rc < 0) return rc;

  size_t szW = (size_t)rows * (size_t)ldW * sizeof(float);
  size_t szx = (size_t)cols * sizeof(float);
  size_t szy = (size_t)rows * sizeof(float);
  size_t szb = (size_t)rows * sizeof(float);

  void *dW=nullptr, *dx=nullptr, *dy=nullptr, *db=nullptr;
  if ((rc = dalloc(szW, &dW)) < 0) return rc;
  if ((rc = dalloc(szx, &dx)) < 0) { dfree(dW); return rc; }
  if ((rc = dalloc(szy, &dy)) < 0) { dfree(dx); dfree(dW); return rc; }
  if (bias) {
    if ((rc = dalloc(szb, &db)) < 0) { dfree(dy); dfree(dx); dfree(dW); return rc; }
  }

  if ((rc = memcpy_imm(dW, W, szW)) < 0) goto CLEANUP;
  if ((rc = memcpy_imm(dx, x, szx)) < 0) goto CLEANUP;
  if (bias)     { if ((rc = memcpy_imm(db, bias, szb)) < 0) goto CLEANUP; }
  if (beta!=0.f){ if ((rc = memcpy_imm(dy, y, szy)) < 0) goto CLEANUP; }

  /* Args: W,x,bias,y,rows,cols,ldW,alpha,beta,act */
  ZE_GUARD(set_arg(g_k_gemv_ba, 0, dW),   "set_arg W failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 1, dx),   "set_arg x failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 2, db),   "set_arg bias failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 3, dy),   "set_arg y failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 4, rows), "set_arg rows failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 5, cols), "set_arg cols failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 6, ldW),  "set_arg ldW failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 7, alpha),"set_arg alpha failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 8, beta), "set_arg beta failed");
  ZE_GUARD(set_arg(g_k_gemv_ba, 9, act),  "set_arg act failed");

  ze_group_count_t gc = { (uint32_t)rows, 1u, 1u };
  ZE_GUARD(zeCommandListAppendLaunchKernel(g_cl, g_k_gemv_ba, &gc, nullptr, 0, nullptr),
           "launch gemv_bias_act_f32 failed");

  if ((rc = memcpy_imm(y, dy, szy)) < 0) goto CLEANUP;

CLEANUP:
  dfree(db); dfree(dy); dfree(dx); dfree(dW);
  return rc;
}

/* ---------------------------- Vector tanh -------------------------------- */

/**
 * @copydoc ie_ze_launch_vec_tanh_f32
 */
int ie_ze_launch_vec_tanh_f32(float*       y,
                              const float* x,
                              int          n) {
  set_err(nullptr);
  if (!x || !y || n <= 0) { set_err("invalid arguments"); return -1; }
  int rc = ensure_module_ready();
  if (rc < 0) return rc;

  size_t sz = (size_t)n * sizeof(float);
  void *dx=nullptr,*dy=nullptr;
  if ((rc = dalloc(sz,&dx)) < 0) return rc;
  if ((rc = dalloc(sz,&dy)) < 0) { dfree(dx); return rc; }

  if ((rc = memcpy_imm(dx, x, sz)) < 0) goto CLEANUP;

  ZE_GUARD(set_arg(g_k_tanh, 0, dy), "set y failed");
  ZE_GUARD(set_arg(g_k_tanh, 1, dx), "set x failed");
  ZE_GUARD(set_arg(g_k_tanh, 2, n ), "set n failed");

  /* 1D grid: ceil(n / 256) groups of 256 threads */
  uint32_t groups = (n + 255u) / 256u;
  if (groups == 0) groups = 1;
  ze_group_count_t gc = { groups, 1u, 1u };

  ZE_GUARD(zeCommandListAppendLaunchKernel(g_cl, g_k_tanh, &gc, nullptr, 0, nullptr),
           "launch vec_tanh_f32 failed");

  if ((rc = memcpy_imm(y, dy, sz)) < 0) goto CLEANUP;

CLEANUP:
  dfree(dy); dfree(dx);
  return rc;
}

/* ------------------------------ Packing ---------------------------------- */

/**
 * @copydoc ie_ze_launch_pack_w_blockedk_f32
 */
int ie_ze_launch_pack_w_blockedk_f32(float*       Wp,
                                     const float* W,
                                     int          rows,
                                     int          cols,
                                     int          ldW,
                                     int          block_k) {
  set_err(nullptr);
  if (!Wp || !W || rows <= 0 || cols <= 0 || ldW < cols || block_k <= 0) {
    set_err("invalid arguments");
    return -1;
  }
  int rc = ensure_module_ready();
  if (rc < 0) return rc;

  size_t szW  = (size_t)rows * (size_t)ldW   * sizeof(float);
  size_t szWp = (size_t)rows * (size_t)cols  * sizeof(float);

  void *dW=nullptr, *dWp=nullptr;
  if ((rc = dalloc(szW,  &dW))  < 0) return rc;
  if ((rc = dalloc(szWp, &dWp)) < 0) { dfree(dW); return rc; }

  if ((rc = memcpy_imm(dW, W, szW)) < 0) goto CLEANUP;

  ZE_GUARD(set_arg(g_k_pack, 0, dWp),     "set Wp failed");
  ZE_GUARD(set_arg(g_k_pack, 1, dW),      "set W  failed");
  ZE_GUARD(set_arg(g_k_pack, 2, rows),    "set rows failed");
  ZE_GUARD(set_arg(g_k_pack, 3, cols),    "set cols failed");
  ZE_GUARD(set_arg(g_k_pack, 4, ldW),     "set ldW failed");
  ZE_GUARD(set_arg(g_k_pack, 5, block_k), "set block_k failed");

  /* 2D grid over (cols, rows) with groups (32,8) → same as CUDA choice */
  uint32_t gx = (cols + 32u - 1u) / 32u;
  uint32_t gy = (rows +  8u - 1u) /  8u;
  ze_group_count_t gc = { gx, gy, 1u };

  ZE_GUARD(zeCommandListAppendLaunchKernel(g_cl, g_k_pack, &gc, nullptr, 0, nullptr),
           "launch pack_w_blockedk_f32 failed");

  if ((rc = memcpy_imm(Wp, dWp, szWp)) < 0) goto CLEANUP;

CLEANUP:
  dfree(dWp); dfree(dW);
  return rc;
}

} /* extern "C" */
