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
    [ "Quick start", "index.html#autotoc_md93", [
      [ "Model format (IEBIN v1)", "index.html#autotoc_md94", null ]
    ] ],
    [ "NEW: INT4 <strong>weight‑only</strong> PTQ path (Q4)", "index.html#autotoc_md96", [
      [ "Pipeline overview", "index.html#autotoc_md97", null ]
    ] ],
    [ "Operational notes", "index.html#autotoc_md99", null ],
    [ "Troubleshooting", "index.html#autotoc_md101", null ],
    [ "What INT4 (weight‑only) means here", "index.html#autotoc_md103", null ],
    [ "See also", "index.html#autotoc_md105", null ],
    [ "INT4 Weight-Only — Addendum (2025-10-24 21:04:23 UTC)", "index.html#autotoc_md107", [
      [ "Summary", "index.html#autotoc_md108", null ],
      [ "Prerequisites", "index.html#autotoc_md109", null ],
      [ "Export to IEBIN with INT4", "index.html#autotoc_md110", null ],
      [ "Run the strict benchmark (CPU)", "index.html#autotoc_md111", null ],
      [ "Run the strict benchmark (CUDA)", "index.html#autotoc_md112", null ],
      [ "Manifest template (example)", "index.html#autotoc_md113", null ],
      [ "Troubleshooting", "index.html#autotoc_md114", null ],
      [ "Notes", "index.html#autotoc_md115", null ]
    ] ],
    [ "Repository Layout", "index.html#autotoc_md116", null ],
    [ "Makefile — Complete Reference", "index.html#autotoc_md117", [
      [ "Common Targets", "index.html#autotoc_md118", null ],
      [ "Environment Variables (consumed by <tt>make bench*</tt> and/or the CLI)", "index.html#autotoc_md119", null ],
      [ "End‑to‑End Examples", "index.html#autotoc_md120", null ],
      [ "Return Codes", "index.html#autotoc_md121", null ]
    ] ],
    [ "RSS Reporting", "index.html#autotoc_md122", [
      [ "RSS Reporting", "index.html#autotoc_md123", null ]
    ] ],
    [ "Update Journal", "index.html#autotoc_md124", null ],
    [ "What’s new — 2025-11-10", "index.html#autotoc_md126", [
      [ "Step 1 — NUMA‑aware topology & thread binding", "index.html#autotoc_md127", null ],
      [ "Step 2 — “Hot” weights replication per socket", "index.html#autotoc_md128", null ],
      [ "Activation precision (INT8 / FP8) — soft hint", "index.html#autotoc_md129", null ],
      [ "Strict timing rule (re‑stated)", "index.html#autotoc_md130", null ],
      [ "Quick run recipes", "index.html#autotoc_md131", null ]
    ] ],
    [ "What's new — Memory Phase (updated 2025-11-12 18:01:19 UTC)", "index.html#autotoc_md132", [
      [ "New tuning knobs (memory/throughput)", "index.html#autotoc_md133", null ],
      [ "Harness sweep (benchmarks)", "index.html#autotoc_md134", null ],
      [ "Monitoring", "index.html#autotoc_md135", null ],
      [ "CUDA/CPU kernels", "index.html#autotoc_md136", null ]
    ] ],
    [ "</blockquote>", "index.html#autotoc_md137", null ],
    [ "What's new — Block-sparse weights (CPU only, 2025‑11‑14)", "index.html#autotoc_md138", [
      [ "New artifacts", "index.html#autotoc_md139", null ],
      [ "Scope and current limitations", "index.html#autotoc_md140", null ]
    ] ],
    [ "Architectural Decision Records", "md_docs_DECISIONS.html", [
      [ "<strong>ADR-0013 (2025-10-24): Adopt INT4 weight-only PTQ & manifest-guided packing</strong>", "md_docs_DECISIONS.html#autotoc_md15", null ],
      [ "Decision: Add INT4 (Weight-Only) Optional Pipeline (2025-10-24 21:04:23 UTC)", "md_docs_DECISIONS.html#autotoc_md17", null ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_DECISIONS.html#autotoc_md19", null ],
      [ "- <strong>Status:</strong> Accepted. Backward‑compatible; default remains FP32 if unset.", "md_docs_DECISIONS.html#autotoc_md21", null ],
      [ "ADR‑0017 (Memory Streaming Heuristics): Prefetch & Non‑Temporal Loads — 2025-11-12 18:01:19 UTC", "md_docs_DECISIONS.html#autotoc_md22", null ],
      [ "ADR‑0018 (Metrics & Reporting): Spatial Metrics in PERFORMANCE.md — 2025-11-12 18:01:19 UTC", "md_docs_DECISIONS.html#autotoc_md24", null ],
      [ "ADR‑0019 (Sparsity): Block‑sparse weights, CPU‑only prototype — 2025‑11‑14 23:00:00 UTC", "md_docs_DECISIONS.html#autotoc_md25", null ]
    ] ],
    [ "Design (CPU baseline + INT4 path)", "md_docs_DESIGN.html", [
      [ "Process and boundaries", "md_docs_DESIGN.html#autotoc_md26", null ],
      [ "API surface (high level)", "md_docs_DESIGN.html#autotoc_md27", null ],
      [ "Hot path layout", "md_docs_DESIGN.html#autotoc_md28", null ],
      [ "Precision modes", "md_docs_DESIGN.html#autotoc_md29", [
        [ "Floating point", "md_docs_DESIGN.html#autotoc_md30", null ],
        [ "INT8 PTQ (reference)", "md_docs_DESIGN.html#autotoc_md31", null ],
        [ "<strong>NEW — INT4 PTQ (weight‑only)</strong>", "md_docs_DESIGN.html#autotoc_md32", null ]
      ] ],
      [ "Threading model", "md_docs_DESIGN.html#autotoc_md33", null ],
      [ "Layout and caching", "md_docs_DESIGN.html#autotoc_md34", null ],
      [ "Metrics", "md_docs_DESIGN.html#autotoc_md35", null ],
      [ "GPU integration (CUDA path)", "md_docs_DESIGN.html#autotoc_md36", null ],
      [ "INT4 Weight-Only Path — Design Addendum (2025-10-24 21:04:23 UTC)", "md_docs_DESIGN.html#autotoc_md38", [
        [ "Goals", "md_docs_DESIGN.html#autotoc_md39", null ],
        [ "Design Choices", "md_docs_DESIGN.html#autotoc_md40", null ],
        [ "Data Flow (INT4 path)", "md_docs_DESIGN.html#autotoc_md41", null ],
        [ "Metrics Integrity", "md_docs_DESIGN.html#autotoc_md42", null ]
      ] ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_DESIGN.html#autotoc_md44", null ],
      [ "Updates — 2025-11-10", "md_docs_DESIGN.html#autotoc_md46", [
        [ "NUMA‑aware topology (<tt>ie_topology</tt>)", "md_docs_DESIGN.html#autotoc_md47", null ],
        [ "Hot weights replication", "md_docs_DESIGN.html#autotoc_md48", null ],
        [ "Activation precision hint", "md_docs_DESIGN.html#autotoc_md49", null ],
        [ "Timing discipline (unchanged semantics)", "md_docs_DESIGN.html#autotoc_md50", null ],
        [ "Example configurations", "md_docs_DESIGN.html#autotoc_md51", null ]
      ] ],
      [ "@code{bash}", "md_docs_DESIGN.html#autotoc_md52", null ],
      [ "Memory Phase Design Addendum (updated 2025-11-12 18:01:19 UTC)", "md_docs_DESIGN.html#autotoc_md53", [
        [ "Goals", "md_docs_DESIGN.html#autotoc_md54", null ],
        [ "Components", "md_docs_DESIGN.html#autotoc_md55", null ],
        [ "Measurement", "md_docs_DESIGN.html#autotoc_md56", null ],
        [ "Backward Compatibility & Fallbacks", "md_docs_DESIGN.html#autotoc_md57", null ],
        [ "Risks & Mitigations", "md_docs_DESIGN.html#autotoc_md58", null ]
      ] ],
      [ "Block‑sparse weights (Phase 2, CPU only)", "md_docs_DESIGN.html#autotoc_md59", [
        [ "Goals", "md_docs_DESIGN.html#autotoc_md60", null ],
        [ "In‑memory layout: <tt>ie_block_sparse_matrix_t</tt>", "md_docs_DESIGN.html#autotoc_md61", null ],
        [ "On‑disk format and loader (<tt>engine/src/sparse_io.c</tt>)", "md_docs_DESIGN.html#autotoc_md62", null ],
        [ "CPU kernel (<tt>engine/src/gemm_block_sparse.c</tt>)", "md_docs_DESIGN.html#autotoc_md63", null ],
        [ "Device abstraction (<tt>engine/src/devices/ie_device_common.c</tt>)", "md_docs_DESIGN.html#autotoc_md64", null ],
        [ "Tools and tests", "md_docs_DESIGN.html#autotoc_md65", [
          [ "Offline converter (<tt>tools/convert_to_block_sparse.c</tt>)", "md_docs_DESIGN.html#autotoc_md66", null ],
          [ "C unit tests (<tt>tests/c/test_block_sparse.c</tt>)", "md_docs_DESIGN.html#autotoc_md67", null ],
          [ "Microbenchmark (<tt>benchmarks/src/microbench_gemv_block_sparse.c</tt>)", "md_docs_DESIGN.html#autotoc_md68", null ]
        ] ],
        [ "Integration strategy and future work", "md_docs_DESIGN.html#autotoc_md69", null ]
      ] ]
    ] ],
    [ "Performance Notes", "md_docs_PERFORMANCE.html", [
      [ "CPU — Summary (latest benchmark)", "md_docs_PERFORMANCE.html#autotoc_md3", null ],
      [ "Latency", "md_docs_PERFORMANCE.html#autotoc_md4", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_PERFORMANCE.html#autotoc_md5", [
        [ "Memory Details", "md_docs_PERFORMANCE.html#autotoc_md6", null ]
      ] ],
      [ "GPU — Summary (latest benchmark)", "md_docs_PERFORMANCE.html#autotoc_md7", null ],
      [ "Latency", "md_docs_PERFORMANCE.html#autotoc_md8", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_PERFORMANCE.html#autotoc_md9", [
        [ "Memory Details", "md_docs_PERFORMANCE.html#autotoc_md10", null ]
      ] ],
      [ "Run Parameters & Conditions", "md_docs_PERFORMANCE.html#autotoc_md11", null ],
      [ "System & Model Info", "md_docs_PERFORMANCE.html#autotoc_md12", null ],
      [ "Comparative Runs", "md_docs_PERFORMANCE.html#autotoc_md13", null ]
    ] ],
    [ "Modules", "modules.html", "modules" ],
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
"functions.html",
"group__ie__int4__quant.html#ga5eab5a797587f75e75af981b851f3d4d",
"ie__kernels__cuda_8h_source.html",
"main__infer_8c.html#a680d5eb9763f52ea89ce1764e01121dc",
"namespaceptq__from__source.html#aa66029bb09d156055e81635a23485ba4",
"spec_8h.html#a7bad93fc007d692d16eeaa767540ccb8",
"structie__device__vtbl.html#af53f74e97e8a3331b3c9edd24a64e668",
"test__dedup__loader_8c_source.html",
"weights__dedup_8h.html#a5579991d9297cadd067b31d66b3005a5"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';