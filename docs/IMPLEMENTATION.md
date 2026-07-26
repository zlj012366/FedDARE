# Implementation

## Code mapping

| Manuscript component | Main implementation |
|---|---|
| Shared semantic encoder and ECA | `src/feddare/models.py` |
| Phase I: Generalization Injection | `src/feddare/local.py` |
| Phase II: Personalized Adaptation | `src/feddare/local.py` |
| Phase III: Behavioral Consistency Evaluation | `src/feddare/behavior.py` |
| Client filtering and encoder aggregation | `src/feddare/federation.py` |
| Poisoning attacks | `src/feddare/attacks.py` |
| Personalized TAcc and ASR | `src/feddare/metrics.py` |

## Main implementation details

- The server task mapping uses the shared encoder followed by adaptive average pooling and a fixed linear head. The head is initialized before federated training and is not updated.
- Each client has a fixed-label synthetic knowledge set. Synthetic inputs are updated by the output discrepancy between the virtual and uploaded encoders.
- The discrepancy in Eq. (8) is implemented as mean-squared error between fixed-head logits.
- CNN-1 to CNN-5 follow the layer widths in the manuscript. CIFAR ResNet profiles use a 3x3 stride-1 stem without max pooling.
- A fixed malicious-client set is sampled before training. The integer count is derived from the configured ratio and stored in `partition_manifest.json`.
- The adaptive Scaling implementation uses a linear 40-round ramp and constrains uploaded updates around the recent three-round moving-average trajectory.
- Every run records the resolved configuration, environment, partition, client model assignment, and metrics.

The supplied YAML files are representative executable configurations for the FedDARE core implementation. Comparative baseline and defense code is not included.
