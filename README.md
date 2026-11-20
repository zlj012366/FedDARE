# FedDARE

FedDARE trains personalized client models with a shared semantic encoder and filters poisoned encoder updates through behavioral consistency evaluation.

## Method

Each communication round has three phases:

1. Generalization Injection freezes the shared encoder and updates the personalized model with a weighted sum of the semantic-input and raw-input losses.
2. Personalized Adaptation freezes the personalized model and updates a client-specific copy of the shared encoder with local supervision.
3. Behavioral Consistency Evaluation builds a virtual encoder from the previous global encoder, client-specific distilled knowledge, and SAM-guided optimization. The server retains uploads whose normalized squared deviation does not exceed `gamma`.

Candidate distilled inputs are committed only for retained clients. Rejected clients keep their previous distilled knowledge. The global encoder is unchanged when no upload is retained. Personalized client models are used for inference.

## Shared semantic encoder

For three-channel inputs, the default encoder is:

```text
Conv2d(3, 116, kernel_size=7, stride=1, padding=3, bias=False)
ReLU
Conv2d(116, 3, kernel_size=5, stride=1, padding=2, bias=False)
ECA(kernel_size=3, bias=False)
```

It preserves the input shape and contains 25,755 trainable parameters.

## Installation

Python 3.9 or later is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For GPU execution, install a PyTorch build compatible with the target CUDA runtime.

## Validation

```bash
feddare --config configs/smoke.yaml --dry-run
pytest -q
python scripts/validate_repo.py
feddare --config configs/smoke.yaml --device cpu
```

## Experiments

```bash
feddare --config configs/experiments/cifar10_k50_c20.yaml --device cuda:0
feddare --config configs/experiments/cifar100_k50_c20.yaml --device cuda:0
feddare --config configs/experiments/cifar10_resnet_k50_c20.yaml --device cuda:0
feddare --config configs/experiments/cifar10_scaling_25pct.yaml --device cuda:0
feddare --config configs/experiments/cifar10_adaptive_scaling_25pct.yaml --device cuda:0
```

Override any configuration field with `--set SECTION.KEY=VALUE`.

```bash
feddare \
  --config configs/experiments/cifar10_adaptive_scaling_25pct.yaml \
  --device cuda:0 \
  --set attack.adaptive_max_scale=0.25 \
  --output-dir runs/cifar10_adaptive_beta025
```

Run five seeds and summarize them with:

```bash
python scripts/run_five_seeds.py \
  --config configs/experiments/cifar10_k50_c20.yaml \
  --device cuda:0 \
  --output-root runs/cifar10_k50_c20

python scripts/summarize_runs.py runs/cifar10_k50_c20
```

## Main settings

| Setting | Configuration name | Default |
|---|---|---:|
| Semantic-guidance weight | `semantic_weight` | 0.5 |
| Local-model learning rate | `model_lr` | 0.01 |
| Semantic-encoder learning rate | `encoder_lr` | 0.01 |
| Virtual optimization steps | `virtual_steps` | 5 |
| SAM perturbation radius | `sam_radius` | 0.05 |
| Virtual learning rate | `virtual_lr` | 0.01 |
| Synthetic-input learning rate | `distill_lr` | 0.01 |
| Behavioral filtering threshold | `threshold` | 0.08 |
| Dirichlet concentration | `dirichlet_alpha` | 0.5 |

Each run writes `resolved_config.json`, `environment.json`, `partition_manifest.json`, and `metrics.csv` to its output directory.

Implementation paths and state-transition rules are listed in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).
