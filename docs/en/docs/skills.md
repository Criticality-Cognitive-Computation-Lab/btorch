# Skills Reference

btorch ships with built-in skills that encode canonical usage patterns for neuromorphic modeling. If you use AI agent, you can invoke these skills explicitly (e.g., "use the btorch-snn-modelling skill") to get vetted, context-aware assistance.

This page summarizes what each skill covers and links to the relevant source files and examples.

## btorch-snn-modelling

**When to invoke it:** Whenever you are building or training spiking neural networks with btorch.

**What it covers:**

- **Stateful modules** — `MemoryModule`, `init_net_state`, `reset_net`, checkpointing
- **The `dt` environment** — `environ.context(dt=...)` usage
- **Training loops** — plain PyTorch and Lightning integration
- **Checkpointing** — saving/loading `memories_rv` with `state_dict()`
- **Truncated BPTT** — `detach_net` for long sequences
- **Common pitfalls** — forgetting `dt`, wrong state names, missing resets

**Key references:**

- Skill source: [`skills/btorch-snn-modelling/SKILL.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/SKILL.md)
- Full training loop: [`skills/btorch-snn-modelling/references/training_example.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/btorch-snn-modelling/references/training_example.md)
- Plain PyTorch example: [`examples/fmnist.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/fmnist.py)
- Lightning example: [`examples/fmnist_lightning.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/examples/fmnist_lightning.py)
- Tests: [`tests/models/test_mem_load_save.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/models/test_mem_load_save.py)

## omegaconf-config

**When to invoke it:** When you need structured configuration, CLI overrides, or launcher-to-worker option forwarding.

**What it covers:**

- **Dataclass-first config** — defaults live in Python, not YAML
- **Composition** — nested dataclasses for common + task-specific settings
- **Variant selection** — dataclass unions with `_type_`
- **Option forwarding** — `to_dotlist` for spawning worker processes
- **Diff utilities** — `diff_conf` for comparing configs

**Key references:**

- Skill source: [`skills/omegaconf-config/SKILL.md`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/skills/omegaconf-config/SKILL.md)
- Utilities: [`btorch/utils/conf.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/btorch/utils/conf.py)
- Tests: [`tests/utils/test_conf.py`](https://github.com/Criticality-Cognitive-Computation-Lab/btorch/blob/main/tests/utils/test_conf.py)
- Guide: [Configuration Guide](guides/configuration.md)

## How to Reference Skills

When prompting an agent, be explicit:

> "Use the btorch-snn-modelling skill to help me write a training loop with truncated BPTT."

> "Use the omegaconf-config skill to set up a batched parameter sweep with launcher-to-worker forwarding."
