from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from runtime_lib.dtypes import dtype_torch

from .datasets import DataConfig, cifar10_loaders, mnist_loaders
from .losses import DistillationLoss
from .models import StudentCNN, TeacherCNN
from .train import TrainConfig, train_student, train_teacher
from .utils import parse_tasks

_logger = logging.getLogger(__name__)


def _resolve_project_path(candidate: str, project_dir: Path) -> Path:
    """Resolve user-provided path against project_dir; tolerant of a leading project folder."""
    raw = Path(candidate)
    if raw.is_absolute():
        return raw
    # If user passed "QRKD/..." while already inside QRKD, strip the prefix
    parts = raw.parts
    if parts and parts[0] == project_dir.name:
        raw = Path(*parts[1:]) if len(parts) > 1 else Path(".")

    candidates = [
        project_dir / raw,
        project_dir.parent / raw,
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fall back to project_dir/raw even if missing; downstream load will error clearly
    return project_dir / raw


def _prepare_loaders(cfg: dict) -> tuple[object, object]:
    dataset = cfg["dataset"]["name"].lower()
    dcfg = DataConfig(
        batch_size=int(cfg["dataset"].get("batch_size", 64)),
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        root=os.path.abspath(os.path.expanduser(cfg["dataset"].get("root", "data"))),
    )
    if dataset == "mnist":
        return mnist_loaders(dcfg)
    if dataset == "cifar10":
        return cifar10_loaders(dcfg)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _logger.info("Saved %s", path)


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    _logger.info("Saved checkpoint: %s", path)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    project_dir = run_dir.parent.parent.resolve()
    torch_dtype = dtype_torch(cfg.get("dtype"))
    device = torch.device(cfg.get("device", "cpu"))
    tasks = parse_tasks(cfg["training"].get("tasks", ["teacher"]))
    _logger.info("Tasks to run: %s", ", ".join(tasks))
    _logger.info("Resolved epoch count: %s", cfg["training"].get("epochs"))

    dataset_name = cfg["dataset"]["name"].lower()
    in_channels = 1 if dataset_name == "mnist" else 3 if dataset_name == "cifar10" else 1
    train_loader, test_loader = _prepare_loaders(cfg)

    teacher = None
    teacher_path = None

    if "teacher" in tasks:
        _logger.info("Training teacher")
        teacher = TeacherCNN(in_channels=in_channels).to(device=device, dtype=torch_dtype)
        tcfg = TrainConfig(
            epochs=int(cfg["training"].get("epochs", 10)),
            lr=float(cfg["training"]["optimizer"].get("lr", 1e-3)),
            device=str(device),
            verbose=True,
            max_batches=None,
        )
        teacher, history = train_teacher(teacher, train_loader, tcfg, test_loader)
        teacher_path = run_dir / "teacher.pt"
        _save_checkpoint(teacher, teacher_path)
        _save_json(run_dir / "history_teacher.json", history)
    else:
        candidate = cfg["training"].get("teacher_path")
        if candidate:
            teacher_path = _resolve_project_path(candidate, project_dir)
            _logger.info("Loading teacher from %s", teacher_path)
            teacher = TeacherCNN(in_channels=in_channels).to(device=device, dtype=torch_dtype)
            teacher.load_state_dict(torch.load(teacher_path, map_location=device))
            teacher.eval()
        else:
            teacher_path = run_dir / "teacher.pt"
            if teacher_path.exists():
                _logger.info("Loading teacher checkpoint from run dir: %s", teacher_path)
                teacher = TeacherCNN().to(device=device, dtype=torch_dtype)
                teacher.load_state_dict(torch.load(teacher_path, map_location=device))
                teacher.eval()

    # Student variants
    qk_weight = float(cfg["training"].get("qkernel_weight", 0.0))
    qk_backend = str(cfg["training"].get("qkernel_backend", "simple"))
    qk_n_modes = cfg["training"].get("qkernel_n_modes")
    qk_n_photons = cfg["training"].get("qkernel_n_photons")
    qk_n_modes = int(qk_n_modes) if qk_n_modes is not None else None
    qk_n_photons = int(qk_n_photons) if qk_n_photons is not None else None

    variants = {
        "student_scratch": DistillationLoss(kd=0.0, dr=0.0, ar=0.0, qk=0.0, qk_backend=qk_backend, qk_n_modes=qk_n_modes, qk_n_photons=qk_n_photons, temperature=float(cfg["training"].get("temperature", 4.0))),
        "student_kd": DistillationLoss(
            kd=float(cfg["training"].get("kd_weight", 0.5)),
            dr=0.0,
            ar=0.0,
            qk=0.0,
            qk_backend=qk_backend,
            qk_n_modes=qk_n_modes,
            qk_n_photons=qk_n_photons,
            temperature=float(cfg["training"].get("temperature", 4.0)),
            kd_alpha=float(cfg["training"].get("kd_alpha", 0.5)),
        ),
        "student_rkd": DistillationLoss(
            kd=0.0,
            dr=float(cfg["training"].get("gamma_distance", 0.1)),
            ar=float(cfg["training"].get("gamma_angle", 0.1)),
            qk=0.0,
            qk_backend=qk_backend,
            qk_n_modes=qk_n_modes,
            qk_n_photons=qk_n_photons,
            temperature=float(cfg["training"].get("temperature", 4.0)),
            kd_alpha=float(cfg["training"].get("kd_alpha", 0.5)),
        ),
        "student_qrkd": DistillationLoss(
            kd=float(cfg["training"].get("kd_weight", 0.5)),
            dr=float(cfg["training"].get("gamma_distance", 0.1)),
            ar=float(cfg["training"].get("gamma_angle", 0.1)),
            qk=qk_weight,
            qk_backend=qk_backend,
            qk_n_modes=qk_n_modes,
            qk_n_photons=qk_n_photons,
            temperature=float(cfg["training"].get("temperature", 4.0)),
            kd_alpha=float(cfg["training"].get("kd_alpha", 0.5)),
        ),
    }

    for task_name, weights in variants.items():
        if task_name not in tasks:
            continue
        if task_name != "student_scratch" and teacher is None:
            raise RuntimeError(f"Task {task_name} requires a teacher; train or provide --training.teacher_path")

        _logger.info("Training %s", task_name)
        student = StudentCNN(in_channels=in_channels).to(device=device, dtype=torch_dtype)
        scfg = TrainConfig(
            epochs=int(cfg["training"].get("epochs", 10)),
            lr=float(cfg["training"]["optimizer"].get("lr", 1e-3)),
            device=str(device),
            verbose=True,
            max_batches=None,
        )
        results = train_student(
            student,
            teacher if task_name != "student_scratch" else None,
            train_loader,
            test_loader,
            scfg,
            weights,
            student_name=task_name,
        )
        _save_checkpoint(student, run_dir / f"{task_name}.pt")
        _save_json(run_dir / f"history_{task_name}.json", results["history"])
        _save_json(run_dir / f"metrics_{task_name}.json", {"test_acc": results["test_acc"]})
