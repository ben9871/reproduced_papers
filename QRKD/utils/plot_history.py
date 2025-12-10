"""Aggregate per-epoch metrics and optionally plot accuracy curves across variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_history(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract(run_dir: Path, stem: str) -> dict[str, list[float]]:
    hist_path = run_dir / f"history_{stem}.json"
    if not hist_path.exists():
        return {}
    return _load_history(hist_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate histories and plot accuracy curves."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-runs", action="store_true", help="Load histories from run directories"
    )
    src.add_argument(
        "--from-json", type=Path, help="Load precomputed combined history JSON"
    )

    p.add_argument("--teacher", type=Path, help="Run dir with history_teacher.json")
    p.add_argument(
        "--scratch", type=Path, help="Run dir with history_student_scratch.json"
    )
    p.add_argument("--kd", type=Path, help="Run dir with history_student_kd.json")
    p.add_argument("--rkd", type=Path, help="Run dir with history_student_rkd.json")
    p.add_argument(
        "--qrkd-simple",
        type=Path,
        help="Run dir with history_student_qrkd.json (simple backend)",
    )
    p.add_argument(
        "--qrkd-merlin",
        type=Path,
        help="Run dir with history_student_qrkd.json (merlin backend)",
    )
    p.add_argument(
        "--qrkd-qiskit",
        type=Path,
        help="Run dir with history_student_qrkd.json (qiskit backend)",
    )

    p.add_argument(
        "--out-json", type=Path, default=Path("results/history_combined.json")
    )
    p.add_argument("--out-plot", type=Path, default=Path("results/accuracy_plot.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    combined: dict[str, dict[str, list[float]]] = {}

    if args.from_json:
        combined = _load_history(args.from_json)
    else:
        runs = {
            "Teacher": (args.teacher, "teacher"),
            "Scratch": (args.scratch, "student_scratch"),
            "KD": (args.kd, "student_kd"),
            "RKD": (args.rkd, "student_rkd"),
            "QRKD-simple": (args.qrkd_simple, "student_qrkd"),
            "QRKD-merlin": (args.qrkd_merlin, "student_qrkd"),
        }
        if args.qrkd_qiskit:
            runs["QRKD-qiskit"] = (args.qrkd_qiskit, "student_qrkd")
        plt.figure(figsize=(8, 5))
        plotted_any = False
        for name, (run_dir, stem) in runs.items():
            if run_dir is None:
                continue
            hist = _extract(run_dir, stem)
            if not hist:
                print(
                    f"[plot_history] Warning: missing history file for {name} at {run_dir}/history_{stem}.json",
                    flush=True,
                )
                continue
            combined[name] = hist
            test_acc = hist.get("test_acc", [])
            if test_acc:
                plt.plot(
                    range(1, len(test_acc) + 1),
                    test_acc,
                    linestyle="-",
                    label=f"{name} test",
                    marker="o",
                )
                plotted_any = True
        if not plotted_any:
            raise SystemExit(
                "No history data found to plot. Ensure history_*.json files exist in the provided run directories."
            )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("QRKD training curves (test accuracy)")
        plt.legend()
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.out_plot)
        plt.close()

    if args.from_json:
        plt.figure(figsize=(8, 5))
        plotted_any = False
        for name, hist in combined.items():
            test_acc = hist.get("test_acc", [])
            if test_acc:
                plt.plot(
                    range(1, len(test_acc) + 1),
                    test_acc,
                    linestyle="-",
                    label=f"{name} test",
                    marker="o",
                )
                plotted_any = True
        if not plotted_any:
            raise SystemExit("No history data found in JSON to plot.")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("QRKD training curves (test accuracy)")
        plt.legend()
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(args.out_plot)
        plt.close()


if __name__ == "__main__":  # pragma: no cover
    main()
