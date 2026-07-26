# FedDARE

**Version: v0.1.0**

PyTorch implementation of **FedDARE: Dual-phase Alignment and Robust Evaluation for Model-Heterogeneous Personalized Federated Learning**.

FedDARE supports semantic collaboration and poisoning-robust aggregation when clients use different local model architectures. Personalized models remain on clients, and only the lightweight shared semantic encoder is communicated.

## Method

FedDARE follows three phases in each communication round:

1. **Generalization Injection**: the frozen shared encoder generates a semantically guided input branch, while the raw-input branch preserves client-specific information. Their weighted losses update the personalized local model.
2. **Personalized Adaptation**: the updated local model is frozen and a client-specific copy of the shared encoder is optimized using local supervision.
3. **Behavioral Consistency Evaluation**: the server uses client-specific distilled synthetic knowledge and SAM-guided virtual optimization to simulate a behaviorally consistent encoder update. Uploaded encoders with deviation no greater than `gamma` are retained and aggregated by local sample size.

The default CIFAR shared encoder is:

```text
7x7 Conv: 3 -> 116
ReLU
5x5 Conv: 116 -> 3
ECA: kernel size 3
```

Both convolutions use stride 1 and same padding. The encoder preserves the input shape and contains 25,874 trainable parameters for three-channel inputs. Only personalized local models are used for inference.

## Included

- FedDARE three-phase training and aggregation
- CNN-1 to CNN-5 and ResNet-18/34/50 client models
- IID and Dirichlet non-IID partitioning
- Client-specific distilled knowledge and SAM virtual optimization
- Label-flip, sign-flip, Gaussian, Scaling, DBA, and adaptive Scaling attacks
- Representative MNIST, CIFAR-10, CIFAR-100, and ResNet configurations
- Five-seed execution, result summarization, tests, and smoke validation

This repository contains the core FedDARE implementation. External baseline and defense implementations used for comparison in the manuscript are not included.

## Structure

```text
FedDARE/
├── configs/
│   ├── smoke.yaml
│   └── paper/
├── docs/
│   └── IMPLEMENTATION.md
├── scripts/
├── src/feddare/
├── tests/
├── LICENSE
├── main.py
└── pyproject.toml
```

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/zlj012366/FedDARE.git
cd FedDARE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the appropriate CUDA-enabled PyTorch build first when GPU execution is required.

## Validation

```bash
feddare --config configs/smoke.yaml --dry-run
pytest -q
python scripts/validate_repo.py
feddare --config configs/smoke.yaml --device cpu
```

The smoke configuration uses generated data and checks the end-to-end execution path.

## Running experiments

```bash
feddare --config configs/paper/cifar10_k50_c20.yaml --device cuda:0
feddare --config configs/paper/cifar100_k50_c20.yaml --device cuda:0
feddare --config configs/paper/cifar10_resnet_k50_c20.yaml --device cuda:0
feddare --config configs/paper/cifar10_scaling_25pct.yaml --device cuda:0
feddare --config configs/paper/cifar10_adaptive_scaling_25pct.yaml --device cuda:0
```

Other adaptive strengths used in the manuscript can be selected through an override:

```bash
feddare --config configs/paper/cifar10_adaptive_scaling_25pct.yaml \
  --device cuda:0 \
  --set attack.adaptive_max_scale=0.25 \
  --output-dir runs/cifar10_adaptive_beta025
```

Any configuration field can be overridden with `--set SECTION.KEY=VALUE`. `python main.py ...` and `feddare ...` are equivalent.

## Main FedDARE settings

| Setting | Default |
|---|---:|
| Semantic-guidance weight `mu` | 0.5 |
| Local-model learning rate `eta_i` | 0.01 |
| Encoder learning rate `eta_a` | 0.01 |
| Virtual steps `M` | 5 |
| SAM radius `rho` | 0.05 |
| Virtual learning rate `alpha_m` | 0.01 |
| Distilled pairs per client | 10 |
| Synthetic-input learning rate `eta_d` | 0.01 |
| Filtering threshold `gamma` | 0.08 |
| Dirichlet parameter `alpha_D` | 0.5 |

## Five-seed evaluation

```bash
python scripts/run_five_seeds.py \
  --config configs/paper/cifar10_k50_c20.yaml \
  --device cuda:0 \
  --output-root runs/cifar10_k50_c20

python scripts/summarize_runs.py runs/cifar10_k50_c20
```

Each run writes `resolved_config.json`, `environment.json`, `partition_manifest.json`, and `metrics.csv` to its output directory. See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the code-to-method mapping and implementation details.

## Citation

```bibtex
@article{zhao_feddare,
  title  = {FedDARE: Dual-phase Alignment and Robust Evaluation for Model-Heterogeneous Personalized Federated Learning},
  author = {Zhao, Lujin and Qin, Sujuan and Shi, Yijie and Li, Wenmin and Gao, Fei and Jin, Zhengping},
  note   = {Manuscript under review}
}
```

## License

FedDARE is released under the [MIT License](LICENSE).
