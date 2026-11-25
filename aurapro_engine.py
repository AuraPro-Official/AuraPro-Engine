"""Local model preparation and inference for AuraPro Engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERSION = "0.0.3"
PORT = 18882
HOST = "127.0.0.1"
sys.path.insert(0, str(ROOT / "terminal"))
from strata_control.pro_catalog import DEFAULT_MODEL, MODELS, SERVE_MODELS


def default_data_dir() -> Path:
    home = Path.home()
    current = home / ".aurapro-engine"
    legacy = home / ".aurapro-pro"
    return legacy if not current.exists() and legacy.is_dir() else current


def configure_environment(data: Path, *, prepare: bool, backend: str) -> None:
    for key in list(os.environ):
        if key.upper().startswith(("STRATA_", "HF_", "HUGGINGFACE_", "HUGGING_FACE_", "MODELSCOPE_", "PYTHON")):
            del os.environ[key]
    os.environ.update({
        "PYTHONNOUSERSITE": "1",
        "STRATA_CONTROL_INSTALL": "1",
        "STRATA_INSTALL_DIR": str(data / "profiles"),
        "STRATA_DATA_DIR": str(data),
        "STRATA_COMPONENTS_DIR": str(data / "components"),
        "STRATA_RUNTIME_DIR": str(data / "runtime"),
        "STRATA_PROFILE_DIR": str(data / "profiles"),
        "STRATA_SETTINGS_FILE": str(data / "settings.json"),
        "STRATA_FEED_MODELS": "0",
        "STRATA_NEED_ENGINE": "1",
        "STRATA_NO_ACQUIRE": "0" if prepare else "1",
    })
    if backend != "auto":
        os.environ["STRATA_ENGINE_BACKEND"] = backend


def profile_path(data: Path, model: str) -> Path:
    family, quant = SERVE_MODELS[model]
    return data / "profiles" / f"strata-{family}-{quant.lower()}.json"


def managed_prepare(root: Path, flags: list[str]) -> int:
    os.environ.update(
        STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(root),
        STRATA_NO_ACQUIRE="0", STRATA_NEED_ENGINE="1",
    )
    import setup

    setup.write_run_script = lambda model, config, *rest, **options: config
    old_argv = sys.argv
    try:
        sys.argv = [str(setup.__file__), *flags]
        return setup.main() or 0
    finally:
        sys.argv = old_argv


def prepare_args(data: Path, args) -> list[str]:
    family, quant = MODELS[args.model]
    return [
        "--setup", "--yes", "--no-start", "--no-browser",
        "--family", family, "--model", quant,
        "--context", str(args.context), "--vision", args.vision,
        "--host", HOST, "--port", str(PORT),
        "--data-dir", str(data), "--models-dir", str(data / "models"),
        *([] if family == "qwen36" else ["--kv", args.kv, "--drafter", "mtp"]),
        *([] if args.backend in ("auto", "metal") else ["--backend", args.backend]),
    ]


def prepare_model(data: Path, args) -> int:
    code = managed_prepare(data / "profiles", prepare_args(data, args))
    if code:
        return code
    config = profile_path(data, args.model)
    if not config.is_file():
        raise RuntimeError("Model preparation completed without writing the expected profile")
    print(f"AuraPro Engine profile: {config}", flush=True)
    return 0


def serve_model(data: Path, args) -> int:
    from serve import server

    config = profile_path(data, args.model)
    if not config.is_file():
        raise RuntimeError(f"{args.model} is not prepared. Run the prepare command first.")
    options = ["--engine", "strata", "--config", str(config), "--port", str(args.port), "--offline"]
    return server.main(options)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="aurapro-engine", description=__doc__)
    parser.add_argument("--version", action="version", version=f"AuraPro Engine {VERSION}")
    parser.add_argument("--data-dir", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List the supported local model presets")
    prepare = commands.add_parser("prepare", help="Download and prepare a selected model")
    serve = commands.add_parser("serve", help="Serve a prepared model on loopback")
    for command in (prepare, serve):
        command.add_argument("--model", choices=MODELS if command is prepare else SERVE_MODELS, default=DEFAULT_MODEL)
        command.add_argument("--backend", choices=("auto", "cuda", "hip", "metal"), default="auto")
    prepare.add_argument("--vision", choices=("yes", "no"), default="yes")
    prepare.add_argument("--context", type=int, choices=(8192, 16384, 32768, 65536, 131072, 262144), default=32768)
    prepare.add_argument("--kv", choices=("int8", "q4_0", "fp16"), default="int8")
    serve.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = default_data_dir()
    if args.command == "serve" and not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "models":
        print(json.dumps([{"id": name, "family": family, "quant": quant, "default": name == DEFAULT_MODEL}
                          for name, (family, quant) in MODELS.items()]))
        return 0
    from strata_control import acquisition, engine_component, targets

    data = args.data_dir.resolve()
    configure_environment(data, prepare=args.command == "prepare", backend=args.backend)
    try:
        target = targets.host_platform()
        backends = {"windows-x86_64": ("cuda", "hip"), "linux-x86_64": ("cuda", "hip"), "macos-arm64": ("metal",)}
        if target not in backends or args.backend not in ("auto", *backends[target]):
            raise ValueError(f"Unsupported engine backend {args.backend} on {target}")
        if target == "macos-arm64" and SERVE_MODELS[args.model][0] == "qwen36":
            raise ValueError("The Strata Metal preparation path does not support the lite presets")
        if args.command == "prepare":
            engine = engine_component.resolve(root=data / "components", full=True)
            if engine.backend not in backends[target]:
                raise ValueError(f"Unsupported engine backend {engine.backend} on {target}")
            args.backend = engine.backend
            os.environ["STRATA_ENGINE_BACKEND"] = engine.backend
            (data / "profiles").mkdir(parents=True, exist_ok=True)
            return prepare_model(data, args)
        return serve_model(data, args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"AuraPro Engine: {exc}", file=sys.stderr)
        return exc.exit_code if isinstance(exc, acquisition.AcquisitionRefused) else 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
