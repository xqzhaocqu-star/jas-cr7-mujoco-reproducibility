# Reproduction protocol

This code export reproduces the registered dual-CR7 MuJoCo comparison. The authoritative settings and 44-trial list are in `reference/protocol.json`. Main trials last 24 s; 24 initial-condition trials last 8 s; two nominal trials halve both integration and control steps. MuJoCo is 3.12.0, the main integrator is RK4, physics timestep 0.000125 s and control timestep 0.0005 s.

The Proposed method retains the original sign feedback, complete tracking-error plus historical composite-learning update, raw-Gram first-IE freezing rule, and parameter projection. Force reactions and true parameters are used only by the plant and offline diagnostics. Nonzero historical residuals are retained. This experiment does not establish zero parameter error or exact sliding. Finite-compliance welds approximate rigid grasps.

Four controlled ablations and an independent two-stage PPC comparator are included. PPC native funnels differ from the common assessment envelope. Failed outcomes must not be dropped. The initial-condition sweep covers one signed six-coordinate direction, not a global fixed-time guarantee.

This export deliberately excludes the private manuscript and old generated data. In the comparison runner and validator, the provenance dependency on the manuscript filename is replaced by this document. All controller and learner implementations, physical models, and fixed settings are unchanged. Existing original-workspace manifests must not be reused as validation for this export: rerun and validate here.

`exponential_identification/` and related scripts are historical diagnostic extensions, NOT the Proposed paper algorithm. They are retained solely to support existing tests and helper imports. Use the comparison entry points in README.md for paper reproduction.

The source environment package versions are in `reference/environment.json`. Identical numerical results across hardware and library versions are not guaranteed; use the included independent validators and half-step trials. The 44 full trials produce approximately 8 GB of output. Runtime depends on CPU and worker count.
