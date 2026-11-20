# Implementation

| Component | Path |
|---|---|
| Shared encoder, ECA, and client models | `src/feddare/models.py` |
| Generalization Injection and Personalized Adaptation | `src/feddare/local.py` |
| Behavioral Consistency Evaluation | `src/feddare/behavior.py` |
| Filtering and weighted encoder aggregation | `src/feddare/federation.py` |
| Upload attacks | `src/feddare/attacks.py` |
| Personalized accuracy and attack success rate | `src/feddare/metrics.py` |

The task head is initialized once with Xavier initialization, shared across clients, fixed for all communication rounds, and excluded from virtual-encoder optimization.

Each client has a synthetic knowledge set initialized from a client-specific seed. Labels are assigned cyclically over class indices and remain fixed. Synthetic inputs are refined from the output discrepancy between the virtual and uploaded encoders.

Behavioral deviation is the squared distance between virtual and uploaded encoder parameters divided by the squared norm of the previous global encoder. A candidate synthetic knowledge update is committed only when the corresponding client upload is retained.

Retained encoders are averaged by local training-set size. If no encoder is retained, the previous global encoder remains unchanged.
