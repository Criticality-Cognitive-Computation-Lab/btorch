---
trigger: always_on
---

1. Use modern python 3.12 typing, e.g. built-in generics (`list`, `dict`) and `|` syntax for unions instead of the `typing` module.
2. When using matplotlib, strictly use the Object-Oriented (OO) interface (e.g., `fig, ax = plt.subplots()`). Avoid state-machine calls like `plt.plot()` or `plt.title()` unless strictly necessary for global configuration.
3. Always write pytest tests for new features under `/tests/`.
4. Generate figures (plots) within tests to allow manual verification of functionality, not just assertions.
5. Use fig_path and save_fig from btorch/utils/file.py and tests/utils/file.py to save fig and other artefacts. The output path will be under fig/<relative_dir>/<test_file_stem>/
6. The tests should also serve as examples if appropriate and documentations with sufficient comments.
7. examples go under examples/, large and long running benchmarks and demos go under benchmark/