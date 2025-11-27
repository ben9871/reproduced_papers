from __future__ import annotations


def default_config() -> dict[str, object]:
    return {
        "seed": 42,
        "outdir": "outdir",
        "device": "cpu",
        "dataset": {"root": "./data", "classes": 2, "batch_size": 128},
        "model": {
            "backend": "classical",  # classical | qiskit | merlin
            "width": 8,
            "loss_dim": 128,
            "batch_norm": False,
            "temperature": 0.07,
            "layers": 2,
            "encoding": "vector",
            "q_ansatz": "sim_circ_14_half",
            "q_sweeps": 1,
            "activation": "null",
            "shots": 100,
            "modes": 10,
            "no_bunching": False,
        },
        "training": {"epochs": 2, "ckpt_step": 1, "le_epochs": 100},
        "logging": {"level": "info"},
    }
