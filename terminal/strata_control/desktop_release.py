"""Install the selected, release-pinned engine into a fresh managed runtime."""

import json
import os
from pathlib import Path

from strata_control import engine_component as ec, feed, longpath, signing, transfer, updater


def selection(root: Path, backend: str):
    items = json.loads((root / "ENGINE-ASSETS.json").read_text(encoding="utf-8"))
    env = dict(os.environ)
    env.pop(ec.ENV_BACKEND, None)
    if backend != "auto":
        env[ec.ENV_BACKEND] = backend
    hardware = updater.detect_hardware(env)
    package, reasons = feed.select_engine([feed._engine_package(item) for item in items], hardware)
    if package is None:
        raise RuntimeError("No compatible AuraPro Engine build: " + "; ".join(reasons))
    return package, hardware


def install(root: Path, backend: str, archive: Path):
    package, hardware = selection(root, backend)
    if archive.stat().st_size != package.size or transfer.sha256_file(archive) != package.sha256:
        raise RuntimeError("AuraPro Engine native archive checksum verification failed")
    components = root / "components"
    ctx = updater.Context(source="", components=components, trusted=signing.load_trusted_keys(),
                          hardware=hardware, state_base=components)
    _, staged = updater.stage_engine(ctx, None, package, archive)
    final = components / "engine" / staged.root.name
    final.parent.mkdir(parents=True, exist_ok=True)
    longpath.long_path(staged.root).rename(longpath.long_path(final))
    component = ec.resolve(components, backend=package.backend, full=False)
    updater.binary_smoke(component, timeout=45)
    return {"version": component.version, "backend": component.backend}
