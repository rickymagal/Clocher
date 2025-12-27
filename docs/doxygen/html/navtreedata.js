/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "Inference Engine (Clocher)", "index.html", [
    [ "Quick start", "index.html#autotoc_md98", [
      [ "Model format (IEBIN v1)", "index.html#autotoc_md99", null ]
    ] ],
    [ "NEW: INT4 <strong>weight‑only</strong> PTQ path (Q4)", "index.html#autotoc_md101", [
      [ "Pipeline overview", "index.html#autotoc_md102", null ]
    ] ],
    [ "Operational notes", "index.html#autotoc_md104", null ],
    [ "Troubleshooting", "index.html#autotoc_md106", null ],
    [ "What INT4 (weight‑only) means here", "index.html#autotoc_md108", null ],
    [ "See also", "index.html#autotoc_md110", null ],
    [ "INT4 Weight-Only — Addendum (2025-10-24 21:04:23 UTC)", "index.html#autotoc_md112", [
      [ "Summary", "index.html#autotoc_md113", null ],
      [ "Prerequisites", "index.html#autotoc_md114", null ],
      [ "Export to IEBIN with INT4", "index.html#autotoc_md115", null ],
      [ "Run the strict benchmark (CPU)", "index.html#autotoc_md116", null ],
      [ "Run the strict benchmark (CUDA)", "index.html#autotoc_md117", null ],
      [ "Manifest template (example)", "index.html#autotoc_md118", null ],
      [ "Troubleshooting", "index.html#autotoc_md119", null ],
      [ "Notes", "index.html#autotoc_md120", null ]
    ] ],
    [ "Repository Layout", "index.html#autotoc_md121", null ],
    [ "Makefile — Complete Reference", "index.html#autotoc_md122", [
      [ "Common Targets", "index.html#autotoc_md123", null ],
      [ "Environment Variables (consumed by <tt>make bench*</tt> and/or the CLI)", "index.html#autotoc_md124", null ],
      [ "End‑to‑End Examples", "index.html#autotoc_md125", null ],
      [ "Return Codes", "index.html#autotoc_md126", null ]
    ] ],
    [ "RSS Reporting", "index.html#autotoc_md127", [
      [ "RSS Reporting", "index.html#autotoc_md128", null ]
    ] ],
    [ "Update Journal", "index.html#autotoc_md129", null ],
    [ "What’s new — 2025-11-10", "index.html#autotoc_md131", [
      [ "Step 1 — NUMA‑aware topology & thread binding", "index.html#autotoc_md132", null ],
      [ "Step 2 — “Hot” weights replication per socket", "index.html#autotoc_md133", null ],
      [ "Activation precision (INT8 / FP8) — soft hint", "index.html#autotoc_md134", null ],
      [ "Strict timing rule (re‑stated)", "index.html#autotoc_md135", null ],
      [ "Quick run recipes", "index.html#autotoc_md136", null ]
    ] ],
    [ "What's new — Memory Phase (updated 2025-11-12 18:01:19 UTC)", "index.html#autotoc_md137", [
      [ "New tuning knobs (memory/throughput)", "index.html#autotoc_md138", null ],
      [ "Harness sweep (benchmarks)", "index.html#autotoc_md139", null ],
      [ "Monitoring", "index.html#autotoc_md140", null ],
      [ "CUDA/CPU kernels", "index.html#autotoc_md141", null ]
    ] ],
    [ "</blockquote>", "index.html#autotoc_md142", null ],
    [ "What's new — Block-sparse weights (CPU only, 2025‑11‑14)", "index.html#autotoc_md143", [
      [ "New artifacts", "index.html#autotoc_md144", null ],
      [ "Scope and current limitations", "index.html#autotoc_md145", null ]
    ] ],
    [ "NEW: Lossless Deduplication (defaults + masks + exceptions)", "index.html#autotoc_md147", [
      [ "1) Generate the three dedup blobs", "index.html#autotoc_md148", null ],
      [ "2) Place/symlink blobs where the runtime loader expects them", "index.html#autotoc_md149", null ],
      [ "3) Strict benchmark with dedup enabled (CPU)", "index.html#autotoc_md150", null ],
      [ "4) Strict benchmark with dedup enabled (CUDA)", "index.html#autotoc_md151", null ]
    ] ],
    [ "Architectural Decision Records", "md_docs_2DECISIONS.html", [
      [ "<strong>ADR-0013 (2025-10-24): Adopt INT4 weight-only PTQ & manifest-guided packing</strong>", "md_docs_2DECISIONS.html#autotoc_md16", null ],
      [ "Decision: Add INT4 (Weight-Only) Optional Pipeline (2025-10-24 21:04:23 UTC)", "md_docs_2DECISIONS.html#autotoc_md18", null ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_2DECISIONS.html#autotoc_md20", null ],
      [ "- <strong>Status:</strong> Accepted. Backward‑compatible; default remains FP32 if unset.", "md_docs_2DECISIONS.html#autotoc_md23", null ],
      [ "ADR‑0017 (Memory Streaming Heuristics): Prefetch & Non‑Temporal Loads — 2025-11-12 18:01:19 UTC", "md_docs_2DECISIONS.html#autotoc_md24", null ],
      [ "ADR‑0018 (Metrics & Reporting): Spatial Metrics in PERFORMANCE.md — 2025-11-12 18:01:19 UTC", "md_docs_2DECISIONS.html#autotoc_md26", null ],
      [ "ADR‑0019 (Sparsity): Block‑sparse weights, CPU‑only prototype — 2025‑11‑14 23:00:00 UTC", "md_docs_2DECISIONS.html#autotoc_md27", null ],
      [ "ADR-0020 (2025-12-22): Adopt lossless dedup artifacts and schema2 runtime loader", "md_docs_2DECISIONS.html#autotoc_md29", null ]
    ] ],
    [ "Design (CPU baseline + INT4 path)", "md_docs_2DESIGN.html", [
      [ "Process and boundaries", "md_docs_2DESIGN.html#autotoc_md30", null ],
      [ "API surface (high level)", "md_docs_2DESIGN.html#autotoc_md31", null ],
      [ "Hot path layout", "md_docs_2DESIGN.html#autotoc_md32", null ],
      [ "Precision modes", "md_docs_2DESIGN.html#autotoc_md33", [
        [ "Floating point", "md_docs_2DESIGN.html#autotoc_md34", null ],
        [ "INT8 PTQ (reference)", "md_docs_2DESIGN.html#autotoc_md35", null ],
        [ "<strong>NEW — INT4 PTQ (weight‑only)</strong>", "md_docs_2DESIGN.html#autotoc_md36", null ]
      ] ],
      [ "Threading model", "md_docs_2DESIGN.html#autotoc_md37", null ],
      [ "Layout and caching", "md_docs_2DESIGN.html#autotoc_md38", null ],
      [ "Metrics", "md_docs_2DESIGN.html#autotoc_md39", null ],
      [ "GPU integration (CUDA path)", "md_docs_2DESIGN.html#autotoc_md40", null ],
      [ "INT4 Weight-Only Path — Design Addendum (2025-10-24 21:04:23 UTC)", "md_docs_2DESIGN.html#autotoc_md42", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md43", null ],
        [ "Design Choices", "md_docs_2DESIGN.html#autotoc_md44", null ],
        [ "Data Flow (INT4 path)", "md_docs_2DESIGN.html#autotoc_md45", null ],
        [ "Metrics Integrity", "md_docs_2DESIGN.html#autotoc_md46", null ]
      ] ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_2DESIGN.html#autotoc_md48", null ],
      [ "Updates — 2025-11-10", "md_docs_2DESIGN.html#autotoc_md50", [
        [ "NUMA‑aware topology (<tt>ie_topology</tt>)", "md_docs_2DESIGN.html#autotoc_md51", null ],
        [ "Hot weights replication", "md_docs_2DESIGN.html#autotoc_md52", null ],
        [ "Activation precision hint", "md_docs_2DESIGN.html#autotoc_md53", null ],
        [ "Timing discipline (unchanged semantics)", "md_docs_2DESIGN.html#autotoc_md54", null ],
        [ "Example configurations", "md_docs_2DESIGN.html#autotoc_md55", null ]
      ] ],
      [ "@icode{bash}", "md_docs_2DESIGN.html#autotoc_md56", null ],
      [ "Memory Phase Design Addendum (updated 2025-11-12 18:01:19 UTC)", "md_docs_2DESIGN.html#autotoc_md57", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md58", null ],
        [ "Components", "md_docs_2DESIGN.html#autotoc_md59", null ],
        [ "Measurement", "md_docs_2DESIGN.html#autotoc_md60", null ],
        [ "Backward Compatibility & Fallbacks", "md_docs_2DESIGN.html#autotoc_md61", null ],
        [ "Risks & Mitigations", "md_docs_2DESIGN.html#autotoc_md62", null ]
      ] ],
      [ "Block‑sparse weights (Phase 2, CPU only)", "md_docs_2DESIGN.html#autotoc_md63", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md64", null ],
        [ "In‑memory layout: <tt>ie_block_sparse_matrix_t</tt>", "md_docs_2DESIGN.html#autotoc_md65", null ],
        [ "On‑disk format and loader (<tt>engine/src/sparse_io.c</tt>)", "md_docs_2DESIGN.html#autotoc_md66", null ],
        [ "CPU kernel (<tt>engine/src/gemm_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md67", null ],
        [ "Device abstraction (<tt>engine/src/devices/ie_device_common.c</tt>)", "md_docs_2DESIGN.html#autotoc_md68", null ],
        [ "Tools and tests", "md_docs_2DESIGN.html#autotoc_md69", [
          [ "Offline converter (<tt>tools/convert_to_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md70", null ],
          [ "C unit tests (<tt>tests/c/test_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md71", null ],
          [ "Microbenchmark (<tt>benchmarks/src/microbench_gemv_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md72", null ]
        ] ],
        [ "Integration strategy and future work", "md_docs_2DESIGN.html#autotoc_md73", null ],
        [ "Lossless Deduplication (Schema2): defaults + masks + exceptions", "md_docs_2DESIGN.html#autotoc_md75", [
          [ "Artifacts", "md_docs_2DESIGN.html#autotoc_md76", null ],
          [ "Offline generation: extract defaults/masks/exceptions", "md_docs_2DESIGN.html#autotoc_md77", null ],
          [ "Runtime controls", "md_docs_2DESIGN.html#autotoc_md78", null ],
          [ "Schema2 JSON compatibility notes (model.ie.json)", "md_docs_2DESIGN.html#autotoc_md79", null ],
          [ "Example strict run (CPU)", "md_docs_2DESIGN.html#autotoc_md80", null ]
        ] ]
      ] ]
    ] ],
    [ "Performance Notes", "md_docs_2PERFORMANCE.html", [
      [ "CPU — Summary (latest benchmark)", "md_docs_2PERFORMANCE.html#autotoc_md2", null ],
      [ "Latency", "md_docs_2PERFORMANCE.html#autotoc_md3", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_2PERFORMANCE.html#autotoc_md4", [
        [ "Memory Details", "md_docs_2PERFORMANCE.html#autotoc_md5", null ],
        [ "Deduplication", "md_docs_2PERFORMANCE.html#autotoc_md6", null ]
      ] ],
      [ "GPU — Summary (latest benchmark)", "md_docs_2PERFORMANCE.html#autotoc_md7", null ],
      [ "Latency", "md_docs_2PERFORMANCE.html#autotoc_md8", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_2PERFORMANCE.html#autotoc_md9", [
        [ "Memory Details", "md_docs_2PERFORMANCE.html#autotoc_md10", null ],
        [ "Deduplication", "md_docs_2PERFORMANCE.html#autotoc_md11", null ]
      ] ],
      [ "Run Parameters & Conditions", "md_docs_2PERFORMANCE.html#autotoc_md12", null ],
      [ "System & Model Info", "md_docs_2PERFORMANCE.html#autotoc_md13", null ],
      [ "Comparative Runs", "md_docs_2PERFORMANCE.html#autotoc_md14", null ]
    ] ],
    [ "Topics", "topics.html", "topics" ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ]
      ] ]
    ] ],
    [ "Data Structures", "annotated.html", [
      [ "Data Structures", "annotated.html", "annotated_dup" ],
      [ "Data Structure Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Data Fields", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", null ],
        [ "Variables", "functions_vars.html", "functions_vars" ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "Globals", "globals.html", [
        [ "All", "globals.html", "globals_dup" ],
        [ "Functions", "globals_func.html", "globals_func" ],
        [ "Variables", "globals_vars.html", null ],
        [ "Typedefs", "globals_type.html", null ],
        [ "Enumerations", "globals_enum.html", null ],
        [ "Enumerator", "globals_eval.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"act__fp8_8c.html",
"dedup__spec_8h.html#a1dbf1873c01aa1c83ff0a1c15bb4ee0f",
"group__IE__GPU__ZE.html#ggab2c254b332e574ed55c2537075337f85a5c3ab96a97ea7aa059b44e7508c28a08",
"ie__kernel__ze_8cpp.html#a20d704d2968105641314c50897df8bdb",
"kv__cache_8c.html",
"namespacededup__prepare__and__extract__all.html#a45246e5fd8e820172df1bd4e9253a791",
"ptq__calib_8py.html#a411585cffe038415fa4a50d31152a8fc",
"structdedup__spec__header__s.html#ad11d8268b50851fe861447d4ede8367b",
"structie__tensor__ref.html#a8d3e19687415a9b972ae0852ff3078e7",
"topology_8c.html#aec68f2b26b90f097bf018395a7cf1bde"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';