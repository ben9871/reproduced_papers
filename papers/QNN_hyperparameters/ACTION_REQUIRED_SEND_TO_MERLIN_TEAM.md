# Action Required: Send to MerLin Team

## Summary

Three items came out of a MerLin 0.4.1 adaptation run on an NVIDIA L4:

- **Bug (High):** a `QuantumLayer` built with `device="cuda"` and
  `MeasurementStrategy.mode_expectations` fails at the first forward pass. Its
  marginalisation mask stays on the CPU.
- **Packaging (High):** `merlinquantum` 0.4.1 does not cap `perceval-quandela`. A
  fresh install pulls Perceval 1.3.x, and `import merlin` then fails.
- **Observation (Low):** identical seeds on CUDA do not give identical training
  runs. Users comparing reruns would benefit from documented guidance on
  determinism.

## Context

The reproduction covers Herbst, De Maio and Brandic, "On Optimizing
Hyperparameters for Quantum Neural Networks" (IEEE QCE 2024, arXiv
2403.18579). It was done in `papers/QNN_hyperparameters/`.

The photonic adaptation is a data-reuploading classifier:
- trainable MZI meshes with angle encoding `pi * x_i` on mode `2i`;
- 2 or 4 photons on 10 or 14 modes, `ComputationSpace.UNBUNCHED`;
- per-mode expectations (`MeasurementStrategy.mode_expectations`) into a linear
  head, trained with Adam on exact gradients.

It ran as a 144-run sweep on a GPU (`merlinquantum` 0.4.1, `perceval-quandela`
1.2.4, torch 2.11.0+cu128, NVIDIA L4, driver 570). The first two items cost
setup time and would block any user running the same pattern on a GPU or in a
fresh environment.

## Items to Send

### High — Measurement / OutputMapper: `mode_expectations` mask stays on the CPU when the layer is built on CUDA

- Type: bug (confirmed in our setup; root cause read from source)
- What happened:
  - We built a `QuantumLayer(..., device="cuda:0")` with
    `measurement_strategy=MeasurementStrategy.mode_expectations(ComputationSpace.UNBUNCHED)`.
  - The forward pass failed with a device-mismatch error in
    `merlin/measurement/mappers.py::marginalize_per_mode`, at
    `probability_distribution @ self.mask.T`.
  - The probabilities are on CUDA, but `layer.measurement_mapping.mask` is on the CPU.
- Evidence:
  - Source, MerLin 0.4.1, `measurement/mappers.py`: the mapper's `__init__`
    builds `keys_tensor = torch.tensor(keys, dtype=torch.long)` and the mask from
    it, then calls `self.register_buffer("mask", mask)`. It accepts a `dtype`
    argument but no `device`, so the mask is always created on the CPU.
    Constructing the layer with `device=` does not move it.
  - Workaround, `papers/QNN_hyperparameters/lib/photonic.py` (lines 78-81):
    `layer.measurement_mapping.mask = layer.measurement_mapping.mask.to(device)`
    after construction. With it, 144/144 GPU runs completed (sweep
    `E-P_photonic_mechanism`, run twice).
  - Found with a GPU feasibility probe on the L4. The source explains why it is
    deterministic for this configuration, but we kept no separate count of failures.
  - Not tested: whether `layer.to("cuda")` after a CPU build moves the mask. It
    is a registered buffer, so it probably would.
- Why it matters: building a layer directly on the GPU with `device=` is the
  natural pattern, and it fails for `mode_expectations` as built. Other mappers
  that create tensors in `__init__` may share the problem; we did not check them.
- Suggested action: pass `device` through to the output mappers and create the
  mask on it, or move child buffers when `QuantumLayer` receives `device`. Add a
  CUDA test for `mode_expectations`.

### High — Packaging: `merlinquantum` 0.4.1 accepts `perceval-quandela` 1.3.x, which breaks `import merlin`

- Type: integration blocker
- What happened: in a fresh virtual environment, `pip install merlinquantum==0.4.1`
  resolved `perceval-quandela` 1.3.1. `import merlin` then raised
  `ModuleNotFoundError: No module named 'perceval.runtime.session'`.
- Evidence:
  - The reproduction's environment setup (Windows, Python 3.12, 2026-10-27).
  - Pinning `perceval-quandela>=1.2,<1.3` (resolved 1.2.4) fixes it locally and on
    the Linux GPU machine.
  - The pin is in `papers/QNN_hyperparameters/requirements.txt`, with the reason
    in a comment.
- Why it matters: anyone installing 0.4.1 without a pin gets a library that fails
  on import, with an error that doesn't point to the version mismatch.
- Suggested action: declare `perceval-quandela<1.3` in the 0.4.x package
  metadata (a patch release or yanked-metadata fix), or make 0.4.x compatible
  with Perceval 1.3.

### Low — Reproducibility on CUDA: same seeds, different training trajectories

- Type: missing documentation (uncertain whether any part is MerLin-specific)
- What happened:
  - We reran the same 144-run sweep with identical code, configs and seeds on
    the same GPU.
  - It reproduced every scientific verdict, but individual effect estimates moved
    by up to 0.032 accuracy.
  - Single-run validation accuracy differed by up to 0.21 (one run on a
    43-sample split).
- Evidence (first attempt vs clean-commit rerun, 144 runs each):
  - Seeded initial phases identical in 144/144 runs.
  - First-step loss equal to within 1e-9 in 141/144.
  - Final loss identical in 72/144, validation accuracy identical in 87/144.
  - The pattern fits float-level CUDA non-determinism that grows over 200 Adam
    steps. We did not test whether it comes from MerLin's kernels, torch's
    defaults (`torch.use_deterministic_algorithms` was not enabled), or both.
- Why it matters: users comparing reruns, or reproducing published MerLin
  numbers on a GPU, may read these differences as bugs.
- Suggested action: document MerLin's determinism behaviour on CUDA in the
  reproducibility notes, including whether
  `torch.use_deterministic_algorithms(True)` is supported. If it's cheap,
  consider a deterministic option.
