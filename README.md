# PDEBench Rebuttal Experiments

This repository includes the rebuttal experiments used to analyze prompt sensitivity, library dependence, and the gap between API proficiency and numerical reasoning.

## 1. Template-Guided Numerical-Reasoning Experiment

This setting provides the model with a code template/skeleton so that the model focuses more on numerical reasoning and less on writing DOLFINx boilerplate.

Before running this experiment, install the project dependencies required by the main DOLFINx-based benchmark environment.

Example command:

```bash
python scripts/run_benchmark.py --prompt-variant template-guided --agent $YOUR_MODEL
```

## 2. Cross-Library Experiments

For the cross-library rebuttal experiments, please install the target library by following its official installation page:

- Firedrake: [https://www.firedrakeproject.org/download.html](https://www.firedrakeproject.org/download.html)
- deal.II: [https://www.dealii.org/](https://www.dealii.org/)

After the environment is ready, run the same benchmark entrypoint and choose the solver library with `--solver-library`.

Firedrake example:

```bash
python scripts/run_benchmark.py --solver-library firedrake --agent $YOUR_MODEL
```

deal.II example:

```bash
python scripts/run_benchmark.py --solver-library dealii --agent $YOUR_MODEL
```

## Notes

- `firedrake` and `dealii` are the valid values currently used by `--solver-library`.
- The benchmark entrypoint is `scripts/run_benchmark.py`.
- For cross-library runs, the evaluation protocol remains the same; only the solver backend changes.
