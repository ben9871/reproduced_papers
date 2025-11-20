#!/usr/bin/env python3
"""QLSTM reproduction CLI aligned with repository template.

Features:
- JSON config via --config + CLI overrides
- Structured run directory under base outdir with config snapshot
- Standard logging (console + run.log)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from lib.config import deep_update, default_config, load_config
from lib.dataset import data as data_factory
from lib.model import build_model
from lib.rendering import save_losses_plot, save_pickle, save_simulation_plot
from pathlib import Path
import numpy as np


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(level.lower(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    # reset
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLSTM vs LSTM trainer")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)

    # model/exp overrides
    p.add_argument("--model", choices=["qlstm", "lstm", "qlstm_photonic"], default=None)
    p.add_argument("--generator", default=None, help="sin|damped_shm|logsine|ma_noise|csv")
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--seq-length", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--vqc-depth", type=int, default=None)
    p.add_argument("--use-preencoders", action="store_true", help="Enable 2 additional VQCs to pre-encode x and h (aligns with 6 VQC variant)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--train-split", type=float, default=None)
    p.add_argument("--fmt", choices=["png", "pdf"], default=None)
    p.add_argument(
        "--snapshot-epochs", "--snapshot_epochs",
        type=str,
        default=None,
        help="Comma-separated epochs to save intermediate simulation plots (e.g., '1,15,30,100')",
    )
    p.add_argument(
        "--plot-width", "--plot_width",
        type=float,
        default=None,
        help="Width of simulation plots in inches (default from config)",
    )
    return p


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = default_config()
    if args.config:
        cfg = deep_update(cfg, load_config(Path(args.config)))
    # overrides
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device
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
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.train_split is not None:
        cfg["experiment"]["train_split"] = args.train_split
    if args.fmt is not None:
        cfg["experiment"]["fmt"] = args.fmt
    if getattr(args, "plot_width", None) is not None:
        cfg["experiment"]["plot_width"] = float(args.plot_width)
    if getattr(args, "snapshot_epochs", None):
        try:
            epochs_list = [int(x) for x in str(args.snapshot_epochs).split(",") if str(x).strip()]
        except Exception as e:  # pragma: no cover - defensive
            raise ValueError(f"Invalid --snapshot-epochs format: {args.snapshot_epochs!r}") from e
        cfg["experiment"]["snapshot_epochs"] = epochs_list
    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on env
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    log = logging.getLogger("train")
    set_seed(cfg["seed"])

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
    log.info("Dataset samples=%d (train=%d, test=%d)", len(x), n_train, len(x) - n_train)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    x_train_in = x_train.unsqueeze(2)
    x_test_in = x_test.unsqueeze(2)

    device = torch.device(cfg["device"])
    model = build_model(
        model_cfg["type"], 1, model_cfg["hidden_size"], model_cfg["vqc_depth"], 1,
        use_preencoders=bool(model_cfg.get("use_preencoders", False)),
        shots=int(model_cfg.get("shots", 0)),
        use_photonic_head=bool(model_cfg.get("use_photonic_head", False)),
    ).to(device).double()
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
                full_last = full_pred.transpose(0, 1)[-1].view(-1).detach().cpu().numpy()
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
    # ensure we operate from QLSTM dir
    script_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != script_dir:
        os.chdir(script_dir)

    configure_logging("info")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = resolve_config(args)
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(cfg["outdir"]) / f"run_{timestamp}"
    base_out.mkdir(parents=True, exist_ok=True)

    # reconfigure logs with file handler in run dir
    configure_logging(cfg.get("logging", {}).get("level", "info"), base_out / "run.log")
    logging.getLogger("main").info("Run directory: %s (id=%s)", base_out, base_out.name)

    # config snapshot
    (base_out / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))

    train_and_evaluate(cfg, base_out)
    logging.info("Finished. Artifacts in: %s", base_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
