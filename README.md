# PDEBench Rebuttal Experiments

This repository contains the rebuttal experiments used to analyze:

- prompt sensitivity to numerical hints,
- the gap between API proficiency and numerical reasoning,
- cross-library / cross-language generalization beyond the default DOLFINx path.

## 1. Install from `pyproject.toml`

Install the Python package itself first:

```bash
pip install -e .
```

For the default DOLFINx-based benchmark path, also install the FEM stack separately. One example is:

```bash
conda install -c conda-forge fenics-dolfinx=0.10.0 mpi4py petsc4py
```

## 2. Export Your LLM API Key

Before running experiments, export the API key required by the model/backend you use.

For OpenAI models:

```bash
export OPENAI_API_KEY=YOUR_KEY_HERE
```

Other common examples:

```bash
export ANTHROPIC_API_KEY=YOUR_KEY_HERE
export GOOGLE_API_KEY=YOUR_KEY_HERE
export DASHSCOPE_API_KEY=YOUR_KEY_HERE
```

## 3. Run Rebuttal Experiments

The main entrypoint is:

```bash
python scripts/run_benchmark.py --agent $YOUR_MODEL
```

### 3.1 Template-Guided Numerical-Reasoning Experiment

This setting provides the model with a code template/skeleton so that the model focuses more on numerical reasoning and less on writing DOLFINx boilerplate.

Example command:

```bash
python scripts/run_benchmark.py --prompt-variant template-guided --agent $YOUR_MODEL
```

### 3.2 Prompt-Sensitivity Ablation

In this rebuttal experiment, numerical hints are removed from the prompt. The agent chooses its own discretization, and the final error is compared after bilinear interpolation onto the evaluation grid.

Example command:

```bash
python scripts/run_benchmark.py --agent $YOUR_MODEL
```

### 3.3 Cross-Library Experiments

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

## 4. Rebuttal Experiment Tables

All numbers below are from the rebuttal experiments on `Gemini-3.0-pro`.

### 4.1 Template-Guided / API-Decoupled Experiment

| Setting | Biharmonic | Convection-Diffusion | Heat | Helmholtz | Linear Elasticity | Navier-Stokes | Poisson | Reaction-Diffusion | Stokes | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original paper setting (DOLFINx) | 66.7 | 40.5 | 7.5 | 68.4 | 0.0 | 0.0 | 44.0 | 4.5 | 0.0 | 27.4 |
| Template-guided / API-decoupled | 66.7 | 42.9 | 67.5 | 63.2 | 28.7 | 21.4 | 40.0 | 27.3 | 27.8 | 44.4 |

### 4.2 Cross-Library / Cross-Language Experiment

| Backend / Language | Biharmonic | Convection-Diffusion | Heat | Helmholtz | Linear Elasticity | Navier-Stokes | Poisson | Reaction-Diffusion | Stokes | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DOLFINx (Python) | 66.7 | 40.5 | 7.5 | 68.4 | 0.0 | 0.0 | 44.0 | 4.5 | 0.0 | 27.4 |
| Firedrake (Python) | 66.7 | 40.5 | 75.0 | 57.9 | 14.3 | 3.6 | 28.0 | 54.5 | 11.1 | 41.1 |
| deal.II (C++) | 60.0 | 38.1 | 52.5 | 52.6 | 0.0 | 10.7 | 36.0 | 36.4 | 16.7 | 36.5 |

### 4.3 Prompt-Sensitivity Ablation

This experiment removes numerical hints and even the output-grid specification from the prompt. The agent chooses its own mesh/discretization, and the final error is evaluated after bilinear interpolation to the benchmark grid.

| Setting | Biharmonic | Convection-Diffusion | Heat | Helmholtz | Linear Elasticity | Navier-Stokes | Poisson | Reaction-Diffusion | Stokes | all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original paper setting (DOLFINx) | 66.7 | 40.5 | 7.5 | 68.4 | 0.0 | 0.0 | 44.0 | 4.5 | 0.0 | 27.4 |
| Hint-removed prompt ablation | 53.3 | 33.3 | 62.5 | 68.4 | 25.0 | 17.9 | 34.0 | 27.3 | 16.7 | 37.8 |

## 5. Solver Directory Layout

The cross-library language extensions are organized under `pdebench/solver`.

```text
pdebench/solver/
├── oracle.py
├── common.py
├── _types.py
├── poisson.py
├── heat.py
├── convection_diffusion.py
├── reaction_diffusion.py
├── helmholtz.py
├── biharmonic.py
├── stokes.py
├── navier_stokes.py
├── linear_elasticity.py
├── darcy.py
├── firedrake/
│   ├── __init__.py
│   ├── oracle.py
│   ├── common.py
│   ├── poisson.py
│   ├── heat.py
│   ├── convection_diffusion.py
│   ├── reaction_diffusion.py
│   ├── helmholtz.py
│   ├── biharmonic.py
│   ├── stokes.py
│   ├── navier_stokes.py
│   └── linear_elasticity.py
└── dealii/
    ├── __init__.py
    ├── oracle.py
    ├── common.py
    └── programs/
        ├── CMakeLists.txt
        ├── poisson.cc
        ├── heat.cc
        ├── convection_diffusion.cc
        ├── reaction_diffusion.cc
        ├── helmholtz.cc
        ├── biharmonic.cc
        ├── stokes.cc
        ├── navier_stokes.cc
        ├── linear_elasticity.cc
        └── common/
            ├── case_spec_reader.h
            └── grid_writer.h
```

### Firedrake layout

- `pdebench/solver/firedrake/common.py`: shared Firedrake helper functions
- `pdebench/solver/firedrake/oracle.py`: Firedrake backend dispatcher
- `pdebench/solver/firedrake/*.py`: PDE-family-specific Firedrake baseline solvers

### deal.II layout

- `pdebench/solver/dealii/oracle.py`: Python-side dispatcher for the deal.II backend
- `pdebench/solver/dealii/common.py`: shared utilities for the deal.II backend
- `pdebench/solver/dealii/programs/*.cc`: PDE-family-specific C++ baseline solver programs
- `pdebench/solver/dealii/programs/common/*`: shared C++ utilities for parsing case specs and writing grids

## 6. Notes

- The benchmark entrypoint is `scripts/run_benchmark.py`.
- `firedrake` and `dealii` are the valid values currently used by `--solver-library`.
- For cross-library runs, the evaluation protocol remains the same; only the solver backend changes.
- The `--prompt-variant template-guided` command refers to the rebuttal template-guided experiment setup.
