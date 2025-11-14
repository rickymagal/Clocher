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
    [ "Quick start", "index.html#autotoc_md83", [
      [ "Model format (IEBIN v1)", "index.html#autotoc_md84", null ]
    ] ],
    [ "NEW: INT4 weight‑only PTQ path (Q4)", "index.html#autotoc_md86", [
      [ "Pipeline overview", "index.html#autotoc_md87", null ]
    ] ],
    [ "Operational notes", "index.html#autotoc_md89", null ],
    [ "Troubleshooting", "index.html#autotoc_md91", null ],
    [ "What INT4 (weight‑only) means here", "index.html#autotoc_md93", null ],
    [ "See also", "index.html#autotoc_md95", null ],
    [ "INT4 Weight-Only — Addendum (2025-10-24 21:04:23 UTC)", "index.html#autotoc_md97", [
      [ "Summary", "index.html#autotoc_md98", null ],
      [ "Prerequisites", "index.html#autotoc_md99", null ],
      [ "Export to IEBIN with INT4", "index.html#autotoc_md100", null ],
      [ "Run the strict benchmark (CPU)", "index.html#autotoc_md101", null ],
      [ "Run the strict benchmark (CUDA)", "index.html#autotoc_md102", null ],
      [ "Manifest template (example)", "index.html#autotoc_md103", null ],
      [ "Troubleshooting", "index.html#autotoc_md104", null ],
      [ "Notes", "index.html#autotoc_md105", null ]
    ] ],
    [ "Repository Layout", "index.html#autotoc_md106", null ],
    [ "Makefile — Complete Reference", "index.html#autotoc_md107", [
      [ "Common Targets", "index.html#autotoc_md108", null ],
      [ "Environment Variables (consumed by make bench* and/or the CLI)", "index.html#autotoc_md109", null ],
      [ "End‑to‑End Examples", "index.html#autotoc_md110", null ],
      [ "Return Codes", "index.html#autotoc_md111", null ]
    ] ],
    [ "RSS Reporting", "index.html#autotoc_md112", [
      [ "RSS Reporting", "index.html#autotoc_md113", null ]
    ] ],
    [ "Update Journal", "index.html#autotoc_md114", null ],
    [ "What’s new — 2025-11-10", "index.html#autotoc_md116", [
      [ "Step 1 — NUMA‑aware topology & thread binding", "index.html#autotoc_md117", null ],
      [ "Step 2 — “Hot” weights replication per socket", "index.html#autotoc_md118", null ],
      [ "Activation precision (INT8 / FP8) — soft hint", "index.html#autotoc_md119", null ],
      [ "Strict timing rule (re‑stated)", "index.html#autotoc_md120", null ],
      [ "Quick run recipes", "index.html#autotoc_md121", null ]
    ] ],
    [ "Note: If numactl -H shows a single node, skip numactl --cpunodebind/--membind.", "index.html#autotoc_md122", null ],
    [ "What's new — Memory Phase (updated 2025-11-12 18:01:19 UTC)", "index.html#autotoc_md123", [
      [ "New tuning knobs (memory/throughput)", "index.html#autotoc_md124", null ],
      [ "Harness sweep (benchmarks)", "index.html#autotoc_md125", null ],
      [ "Monitoring", "index.html#autotoc_md126", null ],
      [ "CUDA/CPU kernels", "index.html#autotoc_md127", null ]
    ] ],
    [ "Architectural Decision Records", "md_docs_2DECISIONS.html", [
      [ "ADR-0013 (2025-10-24): Adopt INT4 weight-only PTQ & manifest-guided packing", "md_docs_2DECISIONS.html#autotoc_md4", null ],
      [ "Decision: Add INT4 (Weight-Only) Optional Pipeline (2025-10-24 21:04:23 UTC)", "md_docs_2DECISIONS.html#autotoc_md6", null ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_2DECISIONS.html#autotoc_md8", null ],
      [ "Status: Accepted. Backward‑compatible; default remains FP32 if unset.", "md_docs_2DECISIONS.html#autotoc_md10", null ],
      [ "ADR‑0017 (Memory Streaming Heuristics): Prefetch & Non‑Temporal Loads — 2025-11-12 18:01:19 UTC", "md_docs_2DECISIONS.html#autotoc_md11", null ],
      [ "ADR‑0018 (Metrics & Reporting): Spatial Metrics in PERFORMANCE.md — 2025-11-12 18:01:19 UTC", "md_docs_2DECISIONS.html#autotoc_md13", null ]
    ] ],
    [ "Design (CPU baseline + INT4 path)", "md_docs_2DESIGN.html", [
      [ "Process and boundaries", "md_docs_2DESIGN.html#autotoc_md14", null ],
      [ "API surface (high level)", "md_docs_2DESIGN.html#autotoc_md15", null ],
      [ "Hot path layout", "md_docs_2DESIGN.html#autotoc_md16", null ],
      [ "Precision modes", "md_docs_2DESIGN.html#autotoc_md17", [
        [ "Floating point", "md_docs_2DESIGN.html#autotoc_md18", null ],
        [ "INT8 PTQ (reference)", "md_docs_2DESIGN.html#autotoc_md19", null ],
        [ "NEW — INT4 PTQ (weight‑only)", "md_docs_2DESIGN.html#autotoc_md20", null ]
      ] ],
      [ "Threading model", "md_docs_2DESIGN.html#autotoc_md21", null ],
      [ "Layout and caching", "md_docs_2DESIGN.html#autotoc_md22", null ],
      [ "Metrics", "md_docs_2DESIGN.html#autotoc_md23", null ],
      [ "GPU integration (CUDA path)", "md_docs_2DESIGN.html#autotoc_md24", null ],
      [ "INT4 Weight-Only Path — Design Addendum (2025-10-24 21:04:23 UTC)", "md_docs_2DESIGN.html#autotoc_md26", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md27", null ],
        [ "Design Choices", "md_docs_2DESIGN.html#autotoc_md28", null ],
        [ "Data Flow (INT4 path)", "md_docs_2DESIGN.html#autotoc_md29", null ],
        [ "Metrics Integrity", "md_docs_2DESIGN.html#autotoc_md30", null ]
      ] ],
      [ "Appendix — INT4 (Weight‑Only) Step (Summary)", "md_docs_2DESIGN.html#autotoc_md32", null ],
      [ "Updates — 2025-11-10", "md_docs_2DESIGN.html#autotoc_md34", [
        [ "NUMA‑aware topology (ie_topology)", "md_docs_2DESIGN.html#autotoc_md35", null ],
        [ "Hot weights replication", "md_docs_2DESIGN.html#autotoc_md36", null ],
        [ "Activation precision hint", "md_docs_2DESIGN.html#autotoc_md37", null ],
        [ "Timing discipline (unchanged semantics)", "md_docs_2DESIGN.html#autotoc_md38", null ],
        [ "Example configurations", "md_docs_2DESIGN.html#autotoc_md39", null ]
      ] ],
      [ "autotoc_md40", "md_docs_2DESIGN.html#autotoc_md40", null ],
      [ "Memory Phase Design Addendum (updated 2025-11-12 18:01:19 UTC)", "md_docs_2DESIGN.html#autotoc_md41", [
        [ "Goals", "md_docs_2DESIGN.html#autotoc_md42", null ],
        [ "Components", "md_docs_2DESIGN.html#autotoc_md43", null ],
        [ "Measurement", "md_docs_2DESIGN.html#autotoc_md44", null ],
        [ "Backward Compatibility & Fallbacks", "md_docs_2DESIGN.html#autotoc_md45", null ],
        [ "Risks & Mitigations", "md_docs_2DESIGN.html#autotoc_md46", null ]
      ] ]
    ] ],
    [ "Performance Notes", "md_docs_2PERFORMANCE.html", [
      [ "CPU — Summary (latest benchmark)", "md_docs_2PERFORMANCE.html#autotoc_md51", null ],
      [ "Latency", "md_docs_2PERFORMANCE.html#autotoc_md52", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_2PERFORMANCE.html#autotoc_md53", [
        [ "Memory Details", "md_docs_2PERFORMANCE.html#autotoc_md54", null ]
      ] ],
      [ "GPU — Summary (latest benchmark)", "md_docs_2PERFORMANCE.html#autotoc_md55", null ],
      [ "Latency", "md_docs_2PERFORMANCE.html#autotoc_md56", null ],
      [ "Spatial Complexity (Memory & Cache)", "md_docs_2PERFORMANCE.html#autotoc_md57", [
        [ "Memory Details", "md_docs_2PERFORMANCE.html#autotoc_md58", null ]
      ] ],
      [ "Run Parameters & Conditions", "md_docs_2PERFORMANCE.html#autotoc_md59", null ],
      [ "System & Model Info", "md_docs_2PERFORMANCE.html#autotoc_md60", null ],
      [ "Comparative Runs", "md_docs_2PERFORMANCE.html#autotoc_md61", null ]
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
"globals_func_g.html",
"ie__floatx_8h.html#a1aa580b0b947f7d3a6628e844d9aa852",
"kv__cache_8c.html#a51dbee8312b98b2d1e701dff311671ca",
"namespacetest__harness.html#abe133296ca5ec1fcb012b0876b519c8f",
"structie__config__env__t.html",
"test__weights_8c.html"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';