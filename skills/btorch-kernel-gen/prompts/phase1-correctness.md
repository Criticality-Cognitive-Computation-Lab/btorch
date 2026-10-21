# Phase 1: first correct implementation

Inspect the confirmed Python model and task contract: objective, inputs/outputs, axes/dtypes, initial/final states, units, event order, gradient meaning, representative and complete workloads, validation/evaluation commands and promotion criteria.

Use KernelWiki to research relevant recurrence, sparse connectivity, delay and implementation techniques. Read the installed dependency skills. Write `docs/draft.md` with baseline validation, risks, ranked approaches, concrete steps and evidence. Run `/humanize:gen-plan`, then `/humanize:start-rlcr-loop`.

Produce a simple correct candidate using the implementation approach appropriate to the user. Validate each requested execution path separately. An inference-only task does not require a training/backward implementation; report unrequested paths as NOT_RUN with the reason. Compare matching semantics against the actual Python model, not recreated equations. For a different backward algorithm, use its declared oracle. Record candidate/parent/source versions and results. Resolve representative correctness failures before performance promotion.
