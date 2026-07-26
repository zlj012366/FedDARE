# 🛡️ FedDARE

![Version](https://img.shields.io/badge/version-v0.1.0-blue)
![Python](https://img.shields.io/badge/Python-%E2%89%A53.9-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-implementation-EE4C2C?logo=pytorch&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PyTorch implementation of **FedDARE: Dual-phase Alignment and Robust Evaluation for Model-Heterogeneous Personalized Federated Learning**.

FedDARE supports semantic collaboration and poisoning-robust aggregation when clients use different local model architectures. Personalized models remain on clients, while only the lightweight shared semantic encoder is communicated.

> ℹ️ **Repository scope:** This repository provides the core FedDARE implementation and representative experiment configurations. External baseline and defense implementations used for comparison in the manuscript are not included.

## 🧩 Method

FedDARE consists of three phases in each communication round:

1. **Phase I — Generalization Injection**  
   The frozen shared semantic encoder generates a semantically guided input branch, while the raw-input branch preserves client-specific information. Their weighted supervised losses update the personalized local model.

2. **Phase II — Personalized Adaptation**  
   The updated personalized model is frozen, and a client-specific copy of the shared semantic encoder is optimized using local supervision.

3. **Phase III — Behavioral Consistency Evaluation**  
   The server uses client-specific distilled synthetic knowledge and SAM-guided virtual optimization to simulate a behaviorally consistent encoder update. Uploaded encoders whose deviation does not exceed `gamma` are retained and aggregated according to local sample size.

Only personalized local models are used for inference.

For the correspondence between the manuscript and implementation, see [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## 🧠 Shared Semantic Encoder

The default shared semantic encoder for CIFAR experiments is:

```text
7x7 Conv: 3 -> 116
ReLU
5x5 Conv: 116 -> 3
ECA: kernel size 3
```

Both convolutional layers use stride 1 and same padding. The encoder preserves the input shape and contains **25,874 trainable parameters** for three-channel inputs.

## ✨ Included

- FedDARE three-phase training and aggregation
- Lightweight shared semantic encoder with ECA
- CNN-1 to CNN-5 personalized client models
- ResNet-18, ResNet-34, and ResNet-50 client models
- IID and Dirichlet non-IID data partitioning
- Client-specific distilled synthetic knowledge
- SAM-guided virtual encoder optimization
- Behavioral consistency filtering
- Label-flip, sign-flip, Gaussian, Scaling, DBA, and adaptive Scaling attacks
- Representative MNIST, CIFAR-10, CIFAR-100, and ResNet configurations
- Five-seed execution and result summarization
- Unit tests, repository validation, and smoke testing

## 🗂️ Repository Structure

```text
FedDARE/
├── configs/
│   ├── smoke.yaml
│   └── paper/
├── docs/
│   └── IMPLEMENTATION.md
├── scripts/
├── src/
│   └── feddare/
├── tests/
├── LICENSE
├── main.py
└── pyproject.toml
```

## ⚙️ Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/zlj012366/FedDARE.git
cd FedDARE

python -m venv .venv
```

Activate the virtual environment.

### Linux or macOS

```bash
source .venv/bin/activate
```

### Windows Git Bash

```bash
source .venv/Scripts/activate
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Install FedDARE and the development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For GPU execution, install a CUDA-compatible PyTorch build for the target environment.

## ✅ Validation

Run the following commands from the repository root:

```bash
feddare --config configs/smoke.yaml --dry-run
pytest -q
python scripts/validate_repo.py
feddare --config configs/smoke.yaml --device cpu
```

The smoke configuration uses generated data to validate the end-to-end execution path without downloading an external dataset.

## 🚀 Running Experiments

### Representative configurations

```bash
feddare --config configs/paper/cifar10_k50_c20.yaml --device cuda:0

feddare --config configs/paper/cifar100_k50_c20.yaml --device cuda:0

feddare --config configs/paper/cifar10_resnet_k50_c20.yaml --device cuda:0

feddare --config configs/paper/cifar10_scaling_25pct.yaml --device cuda:0

feddare --config configs/paper/cifar10_adaptive_scaling_25pct.yaml --device cuda:0
```

`python main.py ...` and `feddare ...` use the same execution entry point.

### Configuration overrides

Any configuration field can be overridden with:

```text
--set SECTION.KEY=VALUE
```

For example, the adaptive attack strength can be changed without editing the YAML file:

```bash
feddare \
  --config configs/paper/cifar10_adaptive_scaling_25pct.yaml \
  --device cuda:0 \
  --set attack.adaptive_max_scale=0.25 \
  --output-dir runs/cifar10_adaptive_beta025
```

## 🔧 Main FedDARE Settings

| Setting | Configuration name | Default |
|---|---|---:|
| Semantic-guidance weight | `mu` | 0.5 |
| Local-model learning rate | `eta_i` | 0.01 |
| Semantic-encoder learning rate | `eta_a` | 0.01 |
| Virtual optimization steps | `M` | 5 |
| SAM perturbation radius | `rho` | 0.05 |
| Virtual learning rate | `alpha_m` | 0.01 |
| Distilled pairs per client | `kd_size` | 10 |
| Synthetic-input learning rate | `eta_d` | 0.01 |
| Behavioral filtering threshold | `gamma` | 0.08 |
| Dirichlet concentration parameter | `alpha_D` | 0.5 |

## 🔁 Five-Seed Evaluation

Run five independent seeds:

```bash
python scripts/run_five_seeds.py \
  --config configs/paper/cifar10_k50_c20.yaml \
  --device cuda:0 \
  --output-root runs/cifar10_k50_c20
```

Summarize the generated runs:

```bash
python scripts/summarize_runs.py runs/cifar10_k50_c20
```

## 📊 Output Files

Each run writes the following files to its output directory:

```text
resolved_config.json
environment.json
partition_manifest.json
metrics.csv
```

These files record the resolved configuration, execution environment, client data partition, and round-level evaluation metrics.

Additional implementation details and code-to-method mappings are available in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## 📚 Citation

```bibtex
@article{zhao_feddare,
  title  = {FedDARE: Dual-phase Alignment and Robust Evaluation for Model-Heterogeneous Personalized Federated Learning},
  author = {Zhao, Lujin and Qin, Sujuan and Shi, Yijie and Li, Wenmin and Gao, Fei and Jin, Zhengping},
  note   = {Manuscript under review}
}
```

> This citation entry will be updated once the manuscript is accepted or published.
