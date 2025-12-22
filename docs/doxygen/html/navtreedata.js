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
    [ "Architectural Decision Records", "md_docs_2DECISIONS.html", [
      [ "<strong>ADR-0013 (2025-10-24): Adopt INT4 weight-only PTQ & manifest-guided packing</strong>", "md_docs_2DECISIONS.html#autotoc_md59", null ],
      [ "Decision: Add INT4 (Weight-Only) Optional Pipeline (2025-10-24 21:04:23 UTC)", "md_docs_2DECISIONS.html#autotoc_md61", null ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_2DECISIONS.html#autotoc_md63", null ],
      [ "- <strong>Status:</strong> Accepted. Backward‑compatible; default remains FP32 if unset.", "md_docs_2DECISIONS.html#autotoc_md65", null ],
      [ "ADR‑0017 (Memory Streaming Heuristics): Prefetch & Non‑Temporal Loads — 2025-11-12 18:01:19 UTC", "md_docs_2DECISIONS.html#autotoc_md66", null ],
      [ "ADR‑0018 (Metrics & Reporting): Spatial Metrics in PERFORMANCE.md — 2025-11-12 18:01:19 UTC", "md_docs_2DECISIONS.html#autotoc_md68", null ],
      [ "ADR‑0019 (Sparsity): Block‑sparse weights, CPU‑only prototype — 2025‑11‑14 23:00:00 UTC", "md_docs_2DECISIONS.html#autotoc_md69", null ]
    ] ],
    [ "Design (CPU baseline + INT4 path)", "md_docs_2DESIGN.html", [
      [ "Process and boundaries", "md_docs_2DESIGN.html#autotoc_md13", null ],
      [ "API surface (high level)", "md_docs_2DESIGN.html#autotoc_md14", null ],
      [ "Hot path layout", "md_docs_2DESIGN.html#autotoc_md15", null ],
      [ "Precision modes", "md_docs_2DESIGN.html#autotoc_md16", [
        [ "Floating point", "md_docs_2DESIGN.html#autotoc_md17", null ],
        [ "INT8 PTQ (reference)", "md_docs_2DESIGN.html#autotoc_md18", null ],
        [ "<strong>NEW — INT4 PTQ (weight‑only)</strong>", "md_docs_2DESIGN.html#autotoc_md19", null ]
      ] ],
      [ "Threading model", "md_docs_2DESIGN.html#autotoc_md20", null ],
      [ "Layout and caching", "md_docs_2DESIGN.html#autotoc_md21", null ],
      [ "Metrics", "md_docs_2DESIGN.html#autotoc_md22", null ],
      [ "GPU integration (CUDA path)", "md_docs_2DESIGN.html#autotoc_md23", null ],
      [ "INT4 Weight-Only Path — Design Addendum (2025-10-24 21:04:23 UTC)", "md_docs_2DESIGN.html#autotoc_md25", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md26", null ],
        [ "Design Choices", "md_docs_2DESIGN.html#autotoc_md27", null ],
        [ "Data Flow (INT4 path)", "md_docs_2DESIGN.html#autotoc_md28", null ],
        [ "Metrics Integrity", "md_docs_2DESIGN.html#autotoc_md29", null ]
      ] ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_2DESIGN.html#autotoc_md32", null ],
      [ "Updates — 2025-11-10", "md_docs_2DESIGN.html#autotoc_md34", [
        [ "NUMA‑aware topology (<tt>ie_topology</tt>)", "md_docs_2DESIGN.html#autotoc_md35", null ],
        [ "Hot weights replication", "md_docs_2DESIGN.html#autotoc_md36", null ],
        [ "Activation precision hint", "md_docs_2DESIGN.html#autotoc_md37", null ],
        [ "Timing discipline (unchanged semantics)", "md_docs_2DESIGN.html#autotoc_md38", null ],
        [ "Example configurations", "md_docs_2DESIGN.html#autotoc_md39", null ]
      ] ],
      [ "@icode{bash}", "md_docs_2DESIGN.html#autotoc_md40", null ],
      [ "Memory Phase Design Addendum (updated 2025-11-12 18:01:19 UTC)", "md_docs_2DESIGN.html#autotoc_md41", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md42", null ],
        [ "Components", "md_docs_2DESIGN.html#autotoc_md43", null ],
        [ "Measurement", "md_docs_2DESIGN.html#autotoc_md44", null ],
        [ "Backward Compatibility & Fallbacks", "md_docs_2DESIGN.html#autotoc_md45", null ],
        [ "Risks & Mitigations", "md_docs_2DESIGN.html#autotoc_md46", null ]
      ] ],
      [ "Block‑sparse weights (Phase 2, CPU only)", "md_docs_2DESIGN.html#autotoc_md47", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md48", null ],
        [ "In‑memory layout: <tt>ie_block_sparse_matrix_t</tt>", "md_docs_2DESIGN.html#autotoc_md49", null ],
        [ "On‑disk format and loader (<tt>engine/src/sparse_io.c</tt>)", "md_docs_2DESIGN.html#autotoc_md50", null ],
        [ "CPU kernel (<tt>engine/src/gemm_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md51", null ],
        [ "Device abstraction (<tt>engine/src/devices/ie_device_common.c</tt>)", "md_docs_2DESIGN.html#autotoc_md52", null ],
        [ "Tools and tests", "md_docs_2DESIGN.html#autotoc_md53", [
          [ "Offline converter (<tt>tools/convert_to_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md54", null ],
          [ "C unit tests (<tt>tests/c/test_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md55", null ],
          [ "Microbenchmark (<tt>benchmarks/src/microbench_gemv_block_sparse.c</tt>)", "md_docs_2DESIGN.html#autotoc_md56", null ]
        ] ],
        [ "Integration strategy and future work", "md_docs_2DESIGN.html#autotoc_md57", null ]
      ] ]
    ] ],
    [ "Performance Notes", "md_docs_2PERFORMANCE.html", [
      [ "CPU — Summary (latest benchmark)", "md_docs_2PERFORMANCE.html#autotoc_md1", null ],
      [ "Latency", "md_docs_2PERFORMANCE.html#autotoc_md2", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_2PERFORMANCE.html#autotoc_md3", [
        [ "Memory Details", "md_docs_2PERFORMANCE.html#autotoc_md4", null ]
      ] ],
      [ "GPU — Summary (latest benchmark)", "md_docs_2PERFORMANCE.html#autotoc_md5", null ],
      [ "Latency", "md_docs_2PERFORMANCE.html#autotoc_md6", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_2PERFORMANCE.html#autotoc_md7", [
        [ "Memory Details", "md_docs_2PERFORMANCE.html#autotoc_md8", null ]
      ] ],
      [ "Run Parameters & Conditions", "md_docs_2PERFORMANCE.html#autotoc_md9", null ],
      [ "System & Model Info", "md_docs_2PERFORMANCE.html#autotoc_md10", null ],
      [ "Comparative Runs", "md_docs_2PERFORMANCE.html#autotoc_md11", null ]
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
"ie__kernels_8h.html#a7548a57990692513f283a22543688e1b",
"loader__mmap_8c.html#a751882cdc8516d2e234680d4f89b7d81ace5144926055034bacd1f224c994c336",
"namespacehf__to__iebin__stream.html#aaac90de3ec7303ed1f6912aaf7316080",
"run__benchmark_8sh.html",
"structie__block__sparse__matrix.html#ae07fcb7d88ab31f89fdcc3bcdcdb021f",
"structie__weights__dedup__opts__t.html#a86ae82fb8029f319a275251de5918035",
"util__metrics_8c.html#a115dd2ed1a4c4c9987ad4c3b1e7ed6fa"
];

var SYNCONMSG = 'click to disable panel synchronisation';
var SYNCOFFMSG = 'click to enable panel synchronisation';