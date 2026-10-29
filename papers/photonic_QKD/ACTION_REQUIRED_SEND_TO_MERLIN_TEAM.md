# Action Required: Send to MerLin Team

## Summary

Five items from the PQKD reproduction:

| Priority | Item | Type |
|---|---|---|
| Medium | The PyPI `merlinquantum==0.4.2` wheel reports `__version__ = "0.4.1"` | packaging bug |
| Medium | `probs()` defaults to `UNBUNCHED`, which silently drops bunched outcomes, and the built-in threshold readout was hard to discover | documentation |
| Medium | `shots=` is silently ignored while autograd is enabled | API confusion |
| Low | Full-Fock memory is the binding limit on CPU for 16-mode threshold-detection circuits | improvement request |
| Low | θ = 0 is an exact stationary point of the gradient for beam-splitter meshes | initialisation note |

## Context

The paper reproduced is *Photonic Quantum-Enhanced Knowledge Distillation* (arXiv:2603.14898). It conditions a compressed CNN student on click histograms from a 16-mode loop interferometer, made of nearest-neighbour `pcvl.BS(θ)` with 45 angles and threshold detection.

The reproduction used `QuantumLayer(input_size=0, circuit=..., trainable_parameters=["theta"], measurement_strategy=MeasurementStrategy.probs(ComputationSpace.FOCK))`, with threshold detection applied in post-processing.

The circuits ran at:

| Photons | Fock states | Phase |
|---|---|---|
| 8, input [1,0]×8 | 490,314 | Phase 3 |
| 4, input [1,0,0,0]×4 | 3,876 | Phase 4 |

Training covered two variants:
- sampled SPSA, the paper's method;
- exact autograd through the MerLin probabilities, a simulation-only variant.

The reproduction targets MerLin 0.4.2 (`merlinquantum==0.4.2`); its tests and smoke configurations run on the PyPI release. The `measurement/`, `builder/` and `core/` modules cited below were read in the 0.4.2 wheel.

These items affect any MerLin user who models threshold detectors, reports versions, or trains sampled circuits.

## Items to Send

### Medium — Packaging: the 0.4.2 wheel reports version 0.4.1

- **Type:** bug.
- **What happened:** in the PyPI `merlinquantum==0.4.2` wheel, the dist-info metadata says 0.4.2, but `merlin/__init__.py` line 82 sets `__version__ = "0.4.1"`.
- **Evidence:** read-only inspection of the installed 0.4.2 wheel during the API survey (`papers/photonic_QKD/LOG.md`, "Phase 4 Candidate Directions → MerLin API survey"). Not re-checked against later releases.
- **Why it matters:** a script that logs `merlin.__version__` for provenance records the wrong version. Reproductions in this repository record that value in their run logs and report "MerLin 0.4.2".
- **Suggested action:** derive `__version__` from package metadata (`importlib.metadata.version("merlinquantum")`), or add a release check that the two match.

### Medium — Measurement: the UNBUNCHED default drops bunched outcomes; the threshold readout is hard to find

- **Type:** missing documentation / API confusion.
- **What happened:** `MeasurementStrategy.probs()` uses `ComputationSpace.UNBUNCHED` by default.
  - For 16 modes and 8 photons, the default output has 12,870 entries, against 490,314 in `ComputationSpace.FOCK`.
  - The bunched outcomes are dropped without any warning. For threshold detection, bunched states are real events that must be mapped to click patterns.
  - The reproduction therefore used FOCK plus manual thresholding (clicks = 1[n > 0]).
  - A later source survey found that MerLin already supports this natively, via `probs(FOCK, occupancy_readout=True)` (strategies.py:347-352) or `pcvl.Detector.threshold()` through an `Experiment`. That support was not discoverable from the documentation and examples used.
- **Evidence:**
  - Compute probe in `papers/photonic_QKD/LOG.md` ("Compute Environment → Compute probe"): default output dimension `[1, 12870]` vs FOCK `(1, 490314)`.
  - Implementation in `papers/photonic_QKD/lib/photonic.py` (FOCK, then thresholding).
  - Tests in `papers/photonic_QKD/tests/test_photonic.py`.
- **Why it matters:** a threshold-detector paper reproduced with the default settings silently uses a post-selected distribution, which biases every downstream feature. This is a correctness trap, not just a performance choice.
- **Suggested action:**
  - Document in `probs()` and the measurement guide that UNBUNCHED post-selects out bunched events.
  - Add a "threshold detectors" recipe showing `occupancy_readout=True` / `Detector.threshold()`.
  - Consider a warning when an UNBUNCHED layer is combined with a detector model that expects bunched events.

### Medium — Sampling: `shots=` is silently ignored while autograd is enabled

- **Type:** API confusion.
- **What happened:** calling `layer(x, shots=S, sampling_method=...)` on a `QuantumLayer` returns exact probabilities, not a shot-sampled estimate, whenever gradient tracking is enabled (`measurement/autodiff.py`, lines 51-78). No warning is emitted. The reproduction implemented its own sampling to guarantee shot noise.
- **Evidence:** source reading during the API survey (`papers/photonic_QKD/LOG.md`, MerLin API survey table, row "`layer(x, shots=, sampling_method=)`"). The behaviour looks intentional, so that gradients stay defined; the issue is the silence.
- **Why it matters:** shot-noise studies, as in this paper's Fig. 4, would unknowingly train on exact distributions and report noise-free results as "S-shot" results.
- **Suggested action:**
  - Warn, or raise unless explicitly allowed, when `shots` is passed while grad is enabled.
  - Document the behaviour and the recommended pattern: sample under `torch.no_grad()`, or use a straight-through or parameter-shift estimator.

### Low — Simulation scaling: full-Fock memory is the binding limit for threshold detection on CPU

- **Type:** improvement.
- **What happened:** with FOCK space on a 7.5 GB, CPU-only laptop:

| Configuration | Fock states | Memory per process | Forward + backward |
|---|---|---|---|
| 16 modes, 8 photons | 490,314 | about 1.8 GB | 0.16–0.19 s per step |
| 16 modes, 4 photons | 3,876 | about 0.9 GB | 5.9 ms |
| 16 modes, 12 photons | about 17M | ran out of memory | — |

  - At 8 photons only 2 runs fitted in parallel.
  - The adaptation therefore dropped to 4 photons.
  - The threshold-detection quantity needed (39,203 distinct click patterns at 8 photons) is far smaller than the Fock space it is computed from.
- **Evidence:** `papers/photonic_QKD/LOG.md` (compute probe; "Phase 4 Adaptation Plan" amendments 2026-10-26T16:40Z and T17:20Z with measured times and memory).
- **Why it matters:** threshold-detection circuits with more than about 8–10 photons in 16 or more modes are the regime of interest for photonic hardware models, and they become infeasible to simulate exactly on CPU.
- **Suggested action:**
  - Document memory as a function of (m, n) for FOCK layers.
  - Consider a click-pattern (threshold) computation space or a sampling backend that avoids materialising the full Fock vector.

### Low — Initialisation: θ = 0 is an exact stationary point for beam-splitter meshes

- **Type:** other (documentation note; physics, not a defect).
- **What happened:**
  - With all `pcvl.BS(θ)` angles at 0, the mesh is a fixed permutation and the output distribution is deterministic.
  - In the exact-gradient arm initialised at θ = 0 (N2), the angles stayed exactly at zero for all 30 epochs on all 3 seeds (‖θ‖ = 0.00), while the student trained normally.
  - Training from random θ (N1) moved normally.
- **Evidence:** `papers/photonic_QKD/results/s6_mnist_4ph_complete.md`, row `N2_native_grad_zero`, column ‖θ‖: 0.00 ± 0.00 (n = 3).
- **Why it matters:** users who zero-initialise trainable interferometers, a natural default and the paper's own choice, get a circuit that never trains under gradients and produces no sampling noise. The failure is silent.
- **Suggested action:** add an initialisation note to the `QuantumLayer` / `CircuitBuilder` documentation. Recommend random or small-noise angle initialisation, and point out that the identity is a critical point of the gradient.
