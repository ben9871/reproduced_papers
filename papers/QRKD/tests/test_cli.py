from __future__ import annotations

import pytest
import torch
from common import build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_train_and_evaluate_writes_artifact(tmp_path, monkeypatch):
    from QRKD.lib import runner as qr_runner

    cfg = load_runtime_ready_config()
    cfg["training"]["tasks"] = ["teacher"]
    cfg["training"]["teacher_path"] = None

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

    markers: dict[str, bool] = {}

    def fake_prepare_loaders(_cfg):
        return "train", "test"

    def fake_train_teacher(model, *_):
        markers["teacher"] = True
        return model, {"loss": [0.0], "train_acc": [], "test_acc": []}

    monkeypatch.setattr(qr_runner, "_prepare_loaders", fake_prepare_loaders)
    monkeypatch.setattr(qr_runner, "TeacherCNN", lambda *_, **__: DummyModel())
    monkeypatch.setattr(qr_runner, "train_teacher", fake_train_teacher)

    qr_runner.train_and_evaluate(cfg, tmp_path)

    assert (tmp_path / "teacher.pt").exists()
    assert markers.get("teacher") is True
