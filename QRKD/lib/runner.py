from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

from runtime_lib.dtypes import dtype_torch

from .datasets import DataConfig, mnist_loaders
from .losses import DistillationLoss
from .models import StudentCNN, TeacherCNN
from .train import TrainConfig, train_student, train_teacher
from .utils import parse_tasks

_logger = logging.getLogger(__name__)


def _prepare_loaders(cfg: dict) -> tuple[object, object]:
    dataset = cfg["dataset"]["name"].lower()
    if dataset != "mnist":
        raise ValueError(f"Unsupported dataset: {dataset}")
    dcfg = DataConfig(
        batch_size=int(cfg["dataset"].get("batch_size", 64)),
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        root=os.path.abspath(os.path.expanduser(cfg["dataset"].get("root", "data"))),
    )
    return mnist_loaders(dcfg)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _logger.info("Saved %s", path)


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    _logger.info("Saved checkpoint: %s", path)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    torch_dtype = dtype_torch(cfg.get("dtype"))
    device = torch.device(cfg.get("device", "cpu"))
    tasks = parse_tasks(cfg["training"].get("tasks", ["teacher"]))
    _logger.info("Tasks to run: %s", ", ".join(tasks))

    train_loader, test_loader = _prepare_loaders(cfg)

    teacher = None
    teacher_path = None

    if "teacher" in tasks:
        _logger.info("Training teacher")
        teacher = TeacherCNN().to(device=device, dtype=torch_dtype)
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
            teacher_path = Path(candidate)
            _logger.info("Loading teacher from %s", teacher_path)
            teacher = TeacherCNN().to(device=device, dtype=torch_dtype)
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
    variants = {
        "student_scratch": DistillationLoss(kd=0.0, dr=0.0, ar=0.0, temperature=float(cfg["training"].get("temperature", 4.0))),
        "student_kd": DistillationLoss(
            kd=float(cfg["training"].get("kd_weight", 0.5)),
            dr=0.0,
            ar=0.0,
            temperature=float(cfg["training"].get("temperature", 4.0)),
            kd_alpha=float(cfg["training"].get("kd_alpha", 0.5)),
        ),
        "student_rkd": DistillationLoss(
            kd=0.0,
            dr=float(cfg["training"].get("gamma_distance", 0.1)),
            ar=float(cfg["training"].get("gamma_angle", 0.1)),
            temperature=float(cfg["training"].get("temperature", 4.0)),
            kd_alpha=float(cfg["training"].get("kd_alpha", 0.5)),
        ),
        "student_qrkd": DistillationLoss(
            kd=float(cfg["training"].get("kd_weight", 0.5)),
            dr=float(cfg["training"].get("gamma_distance", 0.1)),
            ar=float(cfg["training"].get("gamma_angle", 0.1)),
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
        student = StudentCNN().to(device=device, dtype=torch_dtype)
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
