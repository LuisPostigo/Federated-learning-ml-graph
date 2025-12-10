# Federated Learning ML Graph

## Contributors

<!-- Contributors -->
<a href="https://github.com/madch3m/Federated-learning-ml-graph/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=madch3m/Federated-learning-ml-graph" />
</a>

---

**Communication-Efficient Deep Learning from Decentralized Data**

This repository contains a complete, extensible implementation of **Federated Averaging (FedAvg)**, **Federated SGD (FedSGD)**, and **Centralized SGD** following the seminal work of McMahan et al. on *Communication-Efficient Learning of Deep Networks from Decentralized Data*. Our implementation is designed as a **research-grade framework** to reproduce and extend experiments on **IID** and **non-IID** data distributions, with modular support for multiple deep learning architectures.

The repository also includes the experimental workflow and results documented in the companion report: **Federated Learning Experiment Report – FedAvg, FedSGD, and Centralized SGD on MNIST** by Long, Postigo, and Dozal.

---

## Table of Contents

1. Overview
2. Technology Stack
3. Repository Structure
4. How to Run
5. Models Supported
6. Dataset Preparation
7. Reproducibility and Environment
8. Cleanup Instructions
9. Citations
10. Contributors

---

## Overview

Federated Learning (FL) enables collaborative training across distributed clients **without centralizing raw data**, preserving privacy while leveraging large-scale heterogeneous datasets. This implementation provides:

* A fully configurable **FedAvg** and **FedSGD** training pipeline
* Support for **IID**, **Dirichlet-partitioned**, and **pathological non-IID** client distributions
* Multiple pretrained baseline architectures
* A Docker-first execution workflow to streamline replication
* Modular Python implementation for research, prototyping, and coursework

The platform has been tested using MNIST and CIFAR-10, with reproducible comparisons across FL algorithms exactly as reported in our experiment manuscript.

---

## Technology Stack

**Languages:** Python 3.10+
**Deep Learning:** PyTorch 2.x, TorchVision 0.17.x
**Containerization:** Docker Engine & Docker Compose
**Data & Utilities:** Pandas 2.2.x, Matplotlib 3.8.x
**OS Support:** macOS, Linux, Windows PowerShell

---

## Repository Structure

```
federated-learning-ml-graph/
│
├── fedavg/                    # Core FedAvg algorithm + client update logic
├── fedsgd/                    # Baseline FedSGD implementation
├── centralized/               # Centralized SGD baseline
│
├── data/                      # Auto-populated datasets (MNIST, CIFAR10)
├── models/                    # CNN, CIFAR10 CNN, ResNet definitions
├── partitioning/              # IID, non-IID, Dirichlet partition utilities
│
├── scripts/
│   ├── setup.sh               # macOS/Linux runner
│   ├── setup.ps1              # Windows runner
│
├── results/                   # Output logs, plots, metrics
└── README.md
```

---

## How to Run

### 1. Using Docker (Recommended)

#### macOS / Linux

```bash
chmod +x setup.sh && ./setup.sh --docker
./setup.sh --docker --model cifar10_cnn
./setup.sh --docker --model resnet
# Short flags:
./setup.sh -d -m cifar10_cnn
```

#### Windows PowerShell

```powershell
.\setup.ps1 --docker
.\setup.ps1 --docker --model cifar10_cnn
```

---

### 2. Using a Local Python Environment

#### macOS / Linux

```bash
chmod +x setup.sh && ./setup.sh --run
./setup.sh --run --model cifar10_cnn

# Or manually:
./setup.sh
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

---

## Models Supported

| Model Key     | Description                                  |
| ------------- | -------------------------------------------- |
| `cnn`         | Baseline CNN for MNIST (default)             |
| `cifar10_cnn` | Deeper CNN optimized for CIFAR-10            |
| `resnet`      | ResNet architecture for extended experiments |

Add your own architecture by dropping a new file into `models/` and registering it in `setup.sh`.

---

## Dataset Preparation

This project automatically downloads all datasets using `torchvision.datasets`:

* **MNIST** — 60,000 train / 10,000 test handwritten digits
* **CIFAR-10** — 50,000 train / 10,000 test RGB images

Partitioning options include:

* **IID** random partition
* **Pathological non-IID** (sorted labels → shards → clients)
* **Dirichlet non-IID** (`--alpha <float>`)

---

## Reproducibility and Environment

**Hardware Tested**

* Intel i7 / Ryzen 7
* Apple M-series
* NVIDIA RTX 3060 (CUDA optional)

**Software**

* Ubuntu 22.04 / macOS 14 / Windows 11
* Python ≥ 3.10
* PyTorch 2.2.x
* TorchVision 0.17.x
* Docker 24+

**Default Hyperparameters**

```
num_clients: 20
sample_clients: 0.25
local_epochs: 2
local_batch_size: 64
rounds: 25
learning_rate: 0.01
momentum: 0.0
weight_decay: 0.0
optimizer: sgd
iid: true/false
dirichlet_alpha: 0.5
seed: 42
device: cpu or cuda
```

---

## Cleanup Instructions

### macOS / Linux

```bash
rm -rf .venv
rm -rf ./data
pip cache purge
rm -rf ~/.cache/torch
rm -rf ~/.cache/matplotlib
```

### Windows PowerShell

```powershell
Remove-Item .venv -Recurse -Force
Remove-Item .\data -Recurse -Force
pip cache purge
Remove-Item "$env:USERPROFILE\.cache\torch" -Recurse -Force
Remove-Item "$env:LOCALAPPDATA\matplotlib" -Recurse -Force
```

---

## Citations

If you use this repository, please cite both the research report and the original FedAvg paper:

**Course Experiment Report**
Long, D., Postigo, L., & Dozal, R. *Federated Learning Experiment Report – FedAvg, FedSGD, and Centralized SGD on MNIST.*

**Original Federated Averaging Paper**
McMahan, B., Moore, E., Ramage, D., et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017.

---

## Contributors

**Core Authors**

* **Luis Postigo** — Systems engineering, model architecture, experiment reproduction
* **Devin Long** — Experiment design, results analysis
* **Raul Dozal** — Data preprocessing, FL configuration design

**Special Thanks**
Course: *COMP 6130 – Data Mining*, Auburn University.

