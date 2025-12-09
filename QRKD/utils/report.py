"""Aggregate QRKD runs and emit a metrics table (train/test/acc gap, T&S gap, distill gain)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _load_history(path: Path) -> Dict[str, List[float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_acc(run_dir: Path, stem: str) -> Tuple[float | None, float | None]:
    """Return (train_acc, test_acc) from history_{stem}.json or metrics_{stem}.json."""
    hist_path = run_dir / f"history_{stem}.json"
    if hist_path.exists():
        hist = _load_history(hist_path)
        train_acc = hist.get("train_acc", [])
        test_acc = hist.get("test_acc", [])
        return (train_acc[-1] if train_acc else None, test_acc[-1] if test_acc else None)
    metrics_path = run_dir / f"metrics_{stem}.json"
    if metrics_path.exists():
        metrics = _load_metrics(metrics_path)
        return (None, metrics.get("test_acc"))
    return (None, None)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def aggregate(runs: Dict[str, Path]) -> List[Dict[str, float | str]]:
    """Compute metrics per variant."""
    teacher_train, teacher_test = _extract_acc(runs["teacher"], "teacher")
    scratch_train, scratch_test = _extract_acc(runs["student_scratch"], "student_scratch")

    variant_specs = [
        ("F. scratch", "student_scratch", "student_scratch"),
        ("KD", "student_kd", "student_kd"),
        ("RKD", "student_rkd", "student_rkd"),
        ("QRKD-simple", "student_qrkd_simple", "student_qrkd"),
        ("QRKD-merlin", "student_qrkd_merlin", "student_qrkd"),
        ("QRKD-qiskit", "student_qrkd_qiskit", "student_qrkd"),
        ("Teacher", "teacher", "teacher"),
    ]

    results = []
    for name, run_key, hist_stem in variant_specs:
        run_dir = runs.get(run_key)
        if run_dir is None:
            continue
        train_acc, test_acc = _extract_acc(run_dir, hist_stem)
        acc_gap = train_acc - test_acc if train_acc is not None and test_acc is not None else None
        ts_gap = (
            teacher_test - test_acc
            if teacher_test is not None and test_acc is not None and name != "Teacher"
            else None
        )
        dist_gain = (
            test_acc - scratch_test
            if scratch_test is not None and test_acc is not None and name not in {"F. scratch", "Teacher"}
            else None
        )
        results.append(
            {
                "name": name,
                "train": train_acc,
                "test": test_acc,
                "acc_gap": acc_gap,
                "ts_gap": ts_gap,
                "dist_gain": dist_gain,
            }
        )
    return results


def print_table(rows: List[Dict[str, float | str]]) -> None:
    headers = ["Variant", "Train", "Test", "Acc. Gap", "T&S Gap", "Dist. Gain"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(
            " | ".join(
                [
                    str(row["name"]),
                    _fmt(row["train"]),
                    _fmt(row["test"]),
                    _fmt(row["acc_gap"]),
                    _fmt(row["ts_gap"]),
                    _fmt(row["dist_gain"]),
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate QRKD runs into a table.")
    p.add_argument("--teacher", type=Path, required=True, help="Run dir with teacher.pt + history_teacher.json")
    p.add_argument("--scratch", type=Path, required=True, help="Run dir with student_scratch history/metrics")
    p.add_argument("--kd", type=Path, required=True, help="Run dir with student_kd history/metrics")
    p.add_argument("--rkd", type=Path, required=True, help="Run dir with student_rkd history/metrics")
    p.add_argument("--qrkd-simple", type=Path, required=True, help="Run dir with student_qrkd history (simple backend)")
    p.add_argument("--qrkd-merlin", type=Path, required=True, help="Run dir with student_qrkd history (merlin backend)")
    p.add_argument("--qrkd-qiskit", type=Path, required=False, help="Run dir with student_qrkd history (qiskit backend)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs = {
        "teacher": args.teacher,
        "student_scratch": args.scratch,
        "student_kd": args.kd,
        "student_rkd": args.rkd,
        "student_qrkd_simple": args.qrkd_simple,
        "student_qrkd_merlin": args.qrkd_merlin,
    }
    if args.qrkd_qiskit:
        runs["student_qrkd_qiskit"] = args.qrkd_qiskit
    rows = aggregate(runs)
    print_table(rows)


if __name__ == "__main__":  # pragma: no cover
    main()
