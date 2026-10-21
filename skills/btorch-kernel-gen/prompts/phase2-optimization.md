# Phase 2: evidence-driven exploration

Start from the best correct Phase 1 candidate. Use KernelWiki research, benchmark results and ncu-report-skill profiles to rank plausible directions by expected benefit and risk. Consider state layout, scheduling, temporal recurrence, event sparsity, connectivity, delay traffic, fusion and recomputation where they fit this task.

For kernels whose cost varies with neuron count, density and activity, use the
[performance sweep](../references/workflow.md#performance-sweep) anchors:
`N={1e3, 1e5}`, `density={0.1%, 5%, 10%}`, target rate `{1, 5, 20 Hz}`.
Run a light initial profile over the sweep grid, then choose 3-5
representative points covering both `N` regimes and the
density/activity operating plane. Re-select those points as the bottleneck changes;
optimize only the current subset, then verify the remaining combinations.
Record `realized_hz` and use one declared baseline. Treat these as profiling
gates inside the Humanize loop; use ncu-report-skill for collection and
diagnosis rather than duplicating its workflow.

Write the ranked directions and concrete subtasks in `docs/draft.md`; run Humanize plan generation and the RLCR implementation/review loop. Use the user's hardware and budget. Choose an iteration budget appropriate to the task.

For each direction, record before/after measurements for the requested inference, training-forward and backward paths, memory, representative correctness and sufficient profile evidence to keep, revise or reject it. Rerun the complete workload set after major improvements. Do not promote a fast failing implementation or quietly change tolerances, oracle or cases.
