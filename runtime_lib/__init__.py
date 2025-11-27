from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable


TYPE_FACTORIES: dict[str, Callable[[Any], Any]] = {}


def _register_default_types() -> None:
    def _csv_int_list(value: str) -> list[int]:
        if isinstance(value, list):  # already parsed (e.g., from tests)
            return [int(v) for v in value]
        text = str(value).strip()
        if not text:
            return []
        result: list[int] = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if chunk:
                result.append(int(chunk))
        return result

    def _path_type(raw: Any) -> Path:
        return Path(raw)

    def _bool_type(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot interpret boolean value from: {raw}")

    TYPE_FACTORIES.update(
        {
            "int": int,
            "float": float,
            "str": str,
            "path": _path_type,
            "csv_int_list": _csv_int_list,
            "bool": _bool_type,
        }
    )


_register_default_types()


def register_cli_type(name: str, factory: Callable[[Any], Any]) -> None:
    TYPE_FACTORIES[name] = factory


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
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
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _purge_project_modules() -> None:
    for name in list(sys.modules):
        if name == "lib" or name.startswith("lib."):
            sys.modules.pop(name, None)


def _ensure_project_on_path(project_dir: Path) -> None:
    project_dir = project_dir.resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    _purge_project_modules()


def _import_callable(path_spec: str) -> Callable[..., Any]:
    module_name, attr = path_spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):  # pragma: no cover
        raise TypeError(f"Callable required: {path_spec}")
    return fn


def _derive_dest(flags: list[str]) -> str:
    for flag in flags:
        if flag.startswith("--"):
            return flag.lstrip("-").replace("-", "_")
    clean = flags[0].lstrip("-")
    return clean.replace("-", "_")


def _set_by_path(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = cfg
    segments = dotted_path.split(".")
    for key in segments[:-1]:
        if not isinstance(cursor.get(key), dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[segments[-1]] = value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def build_cli_parser(cli_schema: dict[str, Any]) -> tuple[argparse.ArgumentParser, list[dict[str, Any]]]:
    parser = argparse.ArgumentParser(
        description=cli_schema.get("description", "Paper reproduction runner"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    processed: list[dict[str, Any]] = []
    for entry in cli_schema.get("arguments", []):
        arg_def = dict(entry)
        flags = arg_def.get("flags")
        if not flags:
            raise ValueError("Each CLI entry requires at least one flag")
        dest = arg_def.get("dest") or _derive_dest(flags)
        arg_def["_dest"] = dest
        kwargs: dict[str, Any] = {"dest": dest}
        for key in ("help", "default", "choices", "metavar", "required", "nargs"):
            if key in arg_def:
                kwargs[key] = arg_def[key]
        action = arg_def.get("action")
        if action:
            kwargs["action"] = action
        else:
            type_name = arg_def.get("type")
            if type_name:
                try:
                    kwargs["type"] = TYPE_FACTORIES[type_name]
                except KeyError as exc:  # pragma: no cover
                    raise ValueError(f"Unsupported CLI type '{type_name}'") from exc
        parser.add_argument(*flags, **kwargs)
        processed.append(arg_def)
    return parser, processed


def _resolve_path(project_dir: Path, invocation_dir: Path, raw: str | Path) -> Path:
    path_value = Path(raw)
    if path_value.is_absolute():
        return path_value

    invocation_candidate = invocation_dir / path_value
    if invocation_candidate.exists():
        return invocation_candidate

    return project_dir / path_value


def apply_cli_overrides(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    arg_defs: list[dict[str, Any]],
    project_dir: Path,
    invocation_dir: Path,
    *,
    load_config: Callable[[Path], dict[str, Any]],
    deep_update: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    for phase in ("config_file", "other"):
        for arg_def in arg_defs:
            apply_spec = arg_def.get("apply") or {}
            kind = apply_spec.get("kind", "config_path")
            if (phase == "config_file" and kind != "config_file") or (
                phase == "other" and kind == "config_file"
            ):
                continue
            dest = arg_def["_dest"]
            value = getattr(args, dest)
            if value is None:
                continue
            action = arg_def.get("action")
            if action == "store_true" and not value:
                continue
            if action == "store_false" and value:
                continue
            if kind == "config_file":
                cfg_path = _resolve_path(project_dir, invocation_dir, value)
                file_cfg = load_config(cfg_path)
                cfg = deep_update(cfg, file_cfg)
                continue
            if kind == "custom_callable":
                fn = _import_callable(apply_spec["callable"])
                cfg = fn(cfg, value, arg_def)
                continue
            dotted_path = apply_spec.get("path")
            if not dotted_path:
                raise ValueError(
                    f"Missing 'path' for config override on CLI flag {arg_def.get('flags')}"
                )
            stored_value = apply_spec.get("value", value)
            _set_by_path(cfg, dotted_path, _json_safe(stored_value))
    return cfg


def load_runtime_meta(project_dir: Path) -> dict[str, Any]:
    meta_path = project_dir / "configs" / "runtime.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing runtime descriptor: {meta_path}")
    meta = _load_json(meta_path)
    required = {"defaults_path", "cli_schema_path", "runner_callable"}
    missing = sorted(required - meta.keys())
    if missing:
        raise KeyError(f"runtime.json missing keys: {', '.join(missing)}")
    return meta


def run_from_project(project_dir: Path, argv: list[str] | None = None) -> Path:
    project_dir = project_dir.resolve()
    invocation_dir = Path.cwd().resolve()
    configure_logging("info")
    if invocation_dir != project_dir:
        logging.info("Switching working directory to %s", project_dir)
        os.chdir(project_dir)
    _ensure_project_on_path(project_dir)

    meta = load_runtime_meta(project_dir)
    cli_schema = _load_json(project_dir / meta["cli_schema_path"])
    parser, arg_defs = build_cli_parser(cli_schema)
    args = parser.parse_args(argv)

    config_module = importlib.import_module("lib.config")
    load_config = getattr(config_module, "load_config")
    deep_update = getattr(config_module, "deep_update")

    defaults_path = project_dir / meta["defaults_path"]
    cfg = load_config(defaults_path)
    cfg = apply_cli_overrides(
        cfg,
        args,
        arg_defs,
        project_dir,
        invocation_dir,
        load_config=load_config,
        deep_update=deep_update,
    )

    seed_callable_path = meta.get("seed_callable")
    seed_value = cfg.get("seed")
    if seed_callable_path and seed_value is not None:
        seed_fn = _import_callable(seed_callable_path)
        seed_fn(seed_value)  # type: ignore[arg-type]

    timestamp_format = meta.get("timestamp_format", "%Y%m%d-%H%M%S")
    timestamp = dt.datetime.now().strftime(timestamp_format)
    run_pattern = meta.get("run_dir_pattern", "run_{timestamp}")
    run_name = run_pattern.format(timestamp=timestamp)

    base_out = Path(cfg.get("outdir", "outdir"))
    run_dir = base_out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_level = cfg.get("logging", {}).get("level", "info") if isinstance(cfg.get("logging"), dict) else "info"
    configure_logging(log_level, run_dir / "run.log")

    snapshot_path = run_dir / "config_snapshot.json"
    snapshot_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    runner_fn = _import_callable(meta["runner_callable"])
    runner_fn(cfg, run_dir)

    logging.info("Finished. Artifacts in: %s", run_dir)
    return run_dir
