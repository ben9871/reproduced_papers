#!/usr/bin/env python3
"""QLSTM reproduction CLI following generic template structure.

Template features retained:
- Load defaults from `configs/defaults.json` via --config layering.
- CLI overrides for common hyperparameters.
- Logging setup (console + run.log) and config snapshot.

Extended QLSTM logic is implemented inside `train_and_evaluate`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from lib.config import deep_update, load_config
from lib.dataset import data as data_factory
from lib.model import build_model
from lib.rendering import save_losses_plot, save_pickle, save_simulation_plot


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
    """Configure root logger similarly to template version."""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(str(level).lower(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    for h in list(root.handlers):  # reset
        root.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLSTM reproduction runner")
    # Template common flags
    p.add_argument("--config", type=str, help="Path to JSON config", default=None)
    p.add_argument("--seed", type=int, help="Random seed", default=None)
    p.add_argument("--outdir", type=str, help="Base output directory", default=None)
    p.add_argument(
        "--device", type=str, help="Device string (cpu, cuda:0, mps)", default=None
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    # Extended QLSTM overrides
    p.add_argument("--model", choices=["qlstm", "lstm", "qlstm_photonic"], default=None)
    p.add_argument(
        "--generator", default=None, help="sin|damped_shm|logsine|ma_noise|csv"
    )
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--vqc-depth", type=int, default=None)
    p.add_argument("--use-preencoders", action="store_true")
    p.add_argument("--train-split", type=float, default=None)
    p.add_argument("--fmt", choices=["png", "pdf"], default=None)
    p.add_argument(
        "--snapshot-epochs",
        type=str,
        default=None,
        help="Comma-separated epochs for intermediate simulation plots",
    )
    p.add_argument("--plot-width", type=float, default=None)
    return p


def resolve_config(args: argparse.Namespace):
    # Start from defaults.json (template style)
    defaults_path = Path("./configs/defaults.json")
    cfg = load_config(defaults_path)
    # Merge optional user config
    if args.config:
        user_cfg = load_config(Path(args.config))
        cfg = deep_update(cfg, user_cfg)
    # Common overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    # Extended overrides
    if args.model is not None:
        cfg["model"]["type"] = args.model
    if args.generator is not None:
        cfg["experiment"]["generator"] = args.generator
    if args.csv_path is not None:
        cfg["experiment"]["csv_path"] = args.csv_path
    if args.seq_length is not None:
        cfg["experiment"]["seq_length"] = args.seq_length
    if args.hidden_size is not None:
        cfg["model"]["hidden_size"] = args.hidden_size
    if args.vqc_depth is not None:
        cfg["model"]["vqc_depth"] = args.vqc_depth
    if args.use_preencoders:
        cfg["model"]["use_preencoders"] = True
    if args.train_split is not None:
        cfg["experiment"]["train_split"] = args.train_split
    if args.fmt is not None:
        cfg["experiment"]["fmt"] = args.fmt
    if args.plot_width is not None:
        cfg["experiment"]["plot_width"] = float(args.plot_width)
    if args.snapshot_epochs:
        try:
            cfg["experiment"]["snapshot_epochs"] = [
                int(x) for x in args.snapshot_epochs.split(",") if x.strip()
            ]
        except Exception as e:
            raise ValueError(
                f"Invalid --snapshot-epochs format: {args.snapshot_epochs!r}"
            ) from e
    return cfg


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(cfg, run_dir: Path) -> None:
    log = logging.getLogger(__name__)
    log.info("Starting training")
    log.debug("Resolved config: %s", json.dumps(cfg, indent=2))
    setup_seed(cfg["seed"])

    exp = cfg["experiment"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    if exp["generator"] == "csv" and exp.get("csv_path"):
        gen = data_factory.get("csv", path=exp["csv_path"])  # type: ignore[arg-type]
    else:
        gen = data_factory.get(exp["generator"])

    dataset_name = getattr(gen, "name", exp["generator"])
    dataset_extra = ""
    if exp["generator"] == "csv" and exp.get("csv_path"):
        dataset_extra = f" (path={exp['csv_path']})"
    log.info(
        "Dataset: %s [key=%s]%s | seq_length=%d train_split=%.2f",
        dataset_name,
        exp["generator"],
        dataset_extra,
        exp["seq_length"],
        exp["train_split"],
    )

    x, y = gen.get_data(exp["seq_length"])  # type: ignore[call-arg]
    n_train = int(exp["train_split"] * len(x))
    log.info(
        "Dataset samples=%d (train=%d, test=%d)", len(x), n_train, len(x) - n_train
    )
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    x_train_in = x_train.unsqueeze(2)
    x_test_in = x_test.unsqueeze(2)

    device = torch.device(cfg["device"])
    model = (
        build_model(
            model_cfg["type"],
            1,
            model_cfg["hidden_size"],
            model_cfg["vqc_depth"],
            1,
            use_preencoders=bool(model_cfg.get("use_preencoders", False)),
            shots=int(model_cfg.get("shots", 0)),
            use_photonic_head=bool(model_cfg.get("use_photonic_head", False)),
        )
        .to(device)
        .double()
    )
    opt = torch.optim.RMSprop(model.parameters(), lr=train_cfg["lr"])
    mse = nn.MSELoss()

    train_losses: list[float] = []
    test_losses: list[float] = []
    best_test: float = float("inf")
    best_epoch: int = 0
    best_train_at_best: float = float("nan")

    snapshot_set = set(exp.get("snapshot_epochs", []) or [])
    # Always include the last epoch in snapshots
    snapshot_set.add(train_cfg["epochs"])
    for epoch in range(train_cfg["epochs"]):
        model.train()
        perm = torch.randperm(x_train_in.size(0))
        batch_losses: list[float] = []
        for i in range(0, x_train_in.size(0), train_cfg["batch_size"]):
            idx = perm[i : i + train_cfg["batch_size"]]
            xb = x_train_in[idx].to(device)
            yb = y_train[idx].to(device)
            pred, _ = model(xb)
            pred_last = pred.transpose(0, 1)[-1].view(-1)
            loss = mse(pred_last, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(float(loss.item()))
        tr_loss = float(sum(batch_losses) / max(1, len(batch_losses)))
        with torch.no_grad():
            model.eval()
            ptest, _ = model(x_test_in.to(device))
            te_loss = mse(ptest.transpose(0, 1)[-1].view(-1), y_test.to(device)).item()
        train_losses.append(tr_loss)
        test_losses.append(float(te_loss))
        ep = epoch + 1
        # Track and save best checkpoint on test loss
        if te_loss < best_test:
            best_test = float(te_loss)
            best_epoch = ep
            best_train_at_best = float(tr_loss)
            # Save best checkpoint
            best_ckpt = run_dir / "model_best.pth"
            torch.save(model.state_dict(), best_ckpt)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info("Epoch %d: train %.6f test %.6f", epoch + 1, tr_loss, te_loss)

        # Save intermediate simulation snapshots if requested
        if ep in snapshot_set:
            log.info("Snapshot Epoch %d: train %.6f test %.6f", ep, tr_loss, te_loss)
            # Update loss curves on snapshot epochs (includes final epoch)
            save_losses_plot(train_losses, str(run_dir), prefix="train")
            save_losses_plot(test_losses, str(run_dir), prefix="test")
            with torch.no_grad():
                model.eval()
                full_pred, _ = model(x.unsqueeze(2).to(device))
                full_last = (
                    full_pred.transpose(0, 1)[-1].view(-1).detach().cpu().numpy()
                )
                save_simulation_plot(
                    y.detach().cpu().numpy(),
                    full_last,
                    str(run_dir),
                    prefix=f"simulation_e{ep}",
                    vline_x=n_train,
                    width=exp.get("plot_width"),
                    title=f"Epoch {ep} — train MSE {tr_loss:.6f}, test MSE {te_loss:.6f}",
                )
                # Save raw data for this snapshot
                np.savez(
                    run_dir / f"simulation_data_e{ep}.npz",
                    y=y.detach().cpu().numpy(),
                    y_pred=full_last,
                    n_train=n_train,
                    epoch=ep,
                    train_mse=tr_loss,
                    test_mse=te_loss,
                )

    # Save checkpoint & histories
    ckpt_path = run_dir / "model_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    save_pickle(train_losses, str(run_dir), "last_TRAINING_LOSS")
    save_pickle(test_losses, str(run_dir), "last_TESTING_LOSS")
    # Persist best-epoch metadata
    try:
        meta_path = run_dir / "best_epoch.txt"
        meta_path.write_text(
            f"best_epoch={best_epoch}\nbest_test_mse={best_test:.8f}\ntrain_mse_at_best={best_train_at_best:.8f}\n",
            encoding="utf-8",
        )
    except Exception as e:  # pragma: no cover
        logging.getLogger("train").warning("Failed to write best_epoch.txt: %s", e)
    # Also export a CSV for convenient inspection
    try:
        csv_path = run_dir / "losses.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_mse,test_mse\n")
            for i, (tr, te) in enumerate(zip(train_losses, test_losses), start=1):
                f.write(f"{i},{tr:.8f},{te:.8f}\n")
        log.info("Wrote losses CSV: %s", csv_path)
    except Exception as e:  # pragma: no cover
        log.warning("Failed to write losses.csv: %s", e)
    log.info("Saved final checkpoint: %s", ckpt_path)


def main(argv: list[str] | None = None) -> int:
    configure_logging("info")  # initial console logging
    script_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != script_dir:
        logging.info("Switching working directory to %s", script_dir)
        os.chdir(script_dir)

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)

    setup_seed(cfg["seed"])
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(cfg["outdir"]) / f"run_{timestamp}"
    base_out.mkdir(parents=True, exist_ok=True)
    configure_logging(cfg.get("logging", {}).get("level", "info"), base_out / "run.log")
    (base_out / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))
    train_and_evaluate(cfg, base_out)
    logging.info("Finished. Artifacts in: %s", base_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
