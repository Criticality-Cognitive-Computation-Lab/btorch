# Phase 2: evidence-driven exploration


Start from the best correct Phase 1 candidate. Use KernelWiki research, benchmark results and ncu-report-skill profiles to rank plausible directions by expected benefit and risk. Consider state layout, scheduling, temporal recurrence, event sparsity, connectivity, delay traffic, fusion and recomputation where they fit this task.

Write the ranked directions and concrete subtasks in `docs/draft.md`; run Humanize plan generation and the RLCR implementation/review loop. Use the user's hardware and budget. Choose an iteration budget appropriate to the task.

For each direction, record before/after measurements for the requested inference, training-forward and backward paths, memory, representative correctness and sufficient profile evidence to keep, revise or reject it. Rerun the complete workload set after major improvements. Do not promote a fast failing implementation or quietly change tolerances, oracle or cases.
