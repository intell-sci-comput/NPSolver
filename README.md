# NPSolver: Neural Poisson Solver with Iterative Physics Supervision

**Official PyTorch Implementation** | KDD 2026

**Authors:** Bocheng Zeng*, Zhang Rui*, Runze Mao, Mengtao Yan, Xuan Bai, Yang Liu, Zhi X. Chen, Hao Sun.

![License](https://img.shields.io/badge/License-MIT-2196F3.svg)
![AI4Science](https://img.shields.io/badge/AI4Science-8A2BE2)
[![arXiv](https://img.shields.io/badge/arXiv-2605.25786-b31b1b.svg)](https://arxiv.org/abs/2605.25786)

> **Abstract**: Efficiently solving Poisson equations on complex, irregular domains remains a fundamental challenge in scientific computing, as classical iterative solvers often suffer from prohibitive runtime due to ill-conditioned systems. While neural operators offer a fast alternative, they typically rely on large-scale labeled datasets or struggle with unstable training dynamics when using physics-informed residual losses. We propose \textsc{NPSolver}, a neural Poisson solver trained without solution labels via iterative physics supervision. Instead of relying on fully converged numerical solutions or raw PDE residuals, \textsc{NPSolver} utilizes a small number of preconditioned conjugate gradient (PCG) steps to refine its own predictions, providing a more stable and well-scaled training signal. Theoretical analysis confirms that this iterative supervision serves as a well-conditioned error proxy and that a stop-gradient design is essential for optimization stability. To better capture boundary-driven features under mixed boundary conditions, we further introduce the Boundary-Aware Transolver (\textsc{BA-Transolver}) architecture that explicitly separates interior and boundary tokenization. Extensive evaluations on 2D and 3D irregular geometries demonstrate that \textsc{NPSolver} outperforms both physics-informed and data-driven baselines. Furthermore, a downstream thermal control task highlights the model's capability for conducting efficient and reliable gradient-based boundary control. We will release our codes and data at https://github.com/intell-sci-comput/NPSolver.

## ✨ Highlights

![network architecture](./assets/model.jpg)

- NPSolver, a label-free neural Poisson solver built on BA-Transolver model and trained with an iterative physics supervision objective.
- BA-Transolver model, which separately tokenizes interior and boundary nodes.
- Theoretical guarantees for iterative physics supervision.

## ✅ Implemented Features

- 2D on corner-removed square: Dirichlet / Neumann / RandomBC
- 3D on cube-with-cylindrical-hole
- Control task on 2D irregular domains

## 📥 Installation

**1. Clone the repository**

```shell
git clone https://github.com/intell-sci-comput/NPSolver.git
cd NPSolver
```

**2. Create an environment and install dependencies**

```shell
conda create -n npsolver python=3.12
conda activate npsolver
pip install -r requirements.txt
```

## 🗂️ Data Preparation

The datasets and pretrained checkpoints used in our experiments are publicly available on Hugging Face:

- Datasets: https://huggingface.co/datasets/bochengz/NPSolver_datasets/tree/main
- Checkpoints: https://huggingface.co/bochengz/NPSolver_models/tree/main

Please download the required files and store them in your local workspace.

## 🧩 Project Structure and Configuration

The current release includes prediction code for both 2D and 3D settings under `predict/`, and the downstream control task under `control/`. The repository is organized as follows:

```text
NPSolver/
├── control/
│   ├── exp_control.py
│   ├── configs/
│   │   ├── config.yaml
│   │   └── exp/
│   │       └── control.yaml
│   └── src/
│       ├── datasets/
│       ├── fvm_residulers/
│       ├── fvm_solvers/
│       ├── models/
│       └── trainers/
├── predict/
│   ├── exp_2d.py
│   ├── exp_3d.py
│   ├── configs/
│   │   ├── config.yaml
│   │   └── exp/
│   │       ├── dirichlet.yaml
│   │       ├── neumann.yaml
│   │       ├── random_bc.yaml
│   │       └── cube.yaml
│   └── src/
│       ├── datasets/
│       ├── fvm_residulers/
│       ├── fvm_solvers/
│       ├── models/
│       └── trainers/
├── assets/
├── requirements.txt
└── README.md
```

The main entry points are `predict/exp_2d.py` (for 2D prediction), `predict/exp_3d.py` (for 3D prediction), and `control/exp_control.py` (for the control task). The project uses Hydra for configuration management:

- `predict/configs/config.yaml` defines the global runtime settings, such as `project.mode`, `project.device`, and `output.path`.
- `predict/configs/exp/*.yaml` defines experiment-specific settings, including dataset paths, boundary-condition types, model hyperparameters, and training parameters, which are the default settings used to reproduce the experiments reported in the paper.
- `control/configs/config.yaml` and `control/configs/exp/control.yaml` define the configuration for the downstream control task.
- The active experiment config is selected through the `defaults` field in each `config.yaml`.

Before running the code, please update the directory-related fields in the config files to match your local environment, especially:

- `data.mesh_dir`
- `data.sol_dir`
- `output.path`

Then set `data.mesh_dir` and `data.sol_dir` to the corresponding local directories, and set `output.path` to the directory where you want checkpoints, logs, and test outputs to be saved.

This project uses SwanLab to upload training logs to the cloud. If you do not want to use SwanLab, you can comment out the related code. The project will still save local logs, checkpoints, and test outputs under `output.path` even if SwanLab is disabled.

## 🚀 Quick Start

After configuring the relevant paths in the config files, you can run prediction experiments from the `predict/` directory:

```shell
cd predict
```

### Predict

#### 2D Experiments

```shell
# Dirichlet
python exp_2d.py exp=dirichlet

# Neumann
python exp_2d.py exp=neumann

# Random BC
python exp_2d.py exp=random_bc
```

#### 3D Experiments

```shell
python exp_3d.py exp=cube
```

The training or evaluation behavior is controlled by `project.mode` in `configs/config.yaml`:

- `project.mode=train` for training
- `project.mode=test` for evaluation

You can also override config values from the command line when needed. For example:

```shell
python exp_2d.py exp=dirichlet project.mode=train
python exp_2d.py exp=neumann project.mode=test
```

During training, `output.name` is typically generated automatically from the current run time and is used to create a unique output directory for logs and checkpoints. During evaluation with `project.mode=test`, please make sure `output.name` is set to the name of the checkpoint you want to load.

To evaluate with our pretrained checkpoints, please first set `output.path` to the directory where the downloaded checkpoints are stored. Then, for the Dirichlet setting, you can use either of the following two ways:

1. Set `output.name=ba-transolver_dirichlet` in the config file, then run:

```shell
python exp_2d.py exp=dirichlet project.mode=test
```

2. Override `output.name` directly from the command line:

```shell
python exp_2d.py exp=dirichlet project.mode=test output.name=ba-transolver_dirichlet
```

### Control

For the downstream control task, switch to the `control/` directory:

```shell
cd control
```

The control code supports three modes through `project.mode` in `control/configs/config.yaml`:

- `project.mode=train` for training the surrogate model used by the control task
- `project.mode=test` for evaluating the surrogate model
- `project.mode=control` for running the boundary control optimization task

You can running the control surrogate model for training, testing, or control as follows:

```shell
python exp_control.py exp=control project.mode=train
python exp_control.py exp=control project.mode=test output.name=ba-transolver_control
python exp_control.py exp=control project.mode=control output.name=ba-transolver_control
```

## 📈 Citation

```bibtex
@inproceedings{npsolver2026,
	author = {Zeng, Bocheng and Zhang, Rui and Mao, Runze and Yan, Mengtao and Bai, Xuan and Liu, Yang and Chen, Zhi X. and Sun, Hao},
	title = {NPSolver: Neural Poisson Solver with Iterative Physics Supervision},
	year = {2026},
	isbn = {9798400722592},
	publisher = {Association for Computing Machinery},
	address = {New York, NY, USA},
	url = {https://doi.org/10.1145/3770855.3818906},
	doi = {10.1145/3770855.3818906},
	booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
	pages = {12772–12783},
	numpages = {12},
	keywords = {poisson equation, neural pde solver, iterative physics supervision, physics-informed machine learning, surrogate modeling},
	location = {Republic of Korea},
	series = {KDD '26}
}
```

## 🤝 Acknowledgement

This work is supported by the Beijing Natural Science Foundation, the National Natural Science Foundation of China, the China Postdoctoral Science Foundation,  and the Postdoctoral Fellowship
Program of CPSF. We thank collaborators from **Renmin University of China**, **Peking University**, **AI for Science Institute**, and **Huawei Technologies Ltd.**
