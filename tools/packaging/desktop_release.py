"""Assemble desktop release assets from this checkout and the pinned upstream builds."""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "https://github.com/Niko1221/Strata/releases/download/v0.2.20/"
FEED_HASH = "15caccf2e43e53c21cd9e03cffbfc79fa918f72abfc5f9481e99aa5de2bb0164"
PLATFORMS = ("windows-x86_64", "linux-x86_64", "macos-arm64")
sys.path.insert(0, str(ROOT))
from aurapro_engine import VERSION


def manifest():
    with urlopen(UPSTREAM + "strata-feed.json", timeout=60) as response:
        data = response.read(65537)
    if hashlib.sha256(data).hexdigest() != FEED_HASH:
        raise RuntimeError("The pinned upstream package list changed")
    feed = json.loads(data)
    controls = {row["platform"]: row for row in feed["control"]["packages"] if row["platform"] in PLATFORMS}
    engines = {row["url"].replace("strata-engine-", "aurapro-engine-native-", 1): row
               for row in feed["engine"]["packages"]
               if row["platform"] in PLATFORMS and row["backend"] in ("cuda", "hip", "metal")}
    if set(controls) != set(PLATFORMS) or len(engines) != 12:
        raise RuntimeError("Incomplete engine release matrix")
    return controls, engines


def download(row, target):
    name = row["url"]
    if Path(name).name != name:
        raise ValueError("Expected a pinned release asset filename")
    digest = hashlib.sha256()
    size = 0
    with urlopen(UPSTREAM + name, timeout=120) as response, target.open("wb") as stream:
        while block := response.read(1 << 20):
            size += len(block)
            if size > row["size"]:
                raise RuntimeError(f"{name}: oversized upstream download")
            digest.update(block)
            stream.write(block)
    if size != row["size"] or digest.hexdigest() != row["sha256"]:
        raise RuntimeError(f"{name}: upstream checksum verification failed")


def control(row, engines, output):
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        bundle = temp / "bundle"
        archive = temp / ("upstream." + row["format"])
        download(row, archive)
        if row["format"] == "zip":
            with zipfile.ZipFile(archive) as src:
                src.extractall(bundle, (member for member in src.infolist() if member.filename.startswith("python/")))
        else:
            with tarfile.open(archive) as src:
                src.extractall(bundle, (member for member in src if member.name.startswith("python/")), filter="data")
        if not ((bundle / "python/python.exe").is_file() or (bundle / "python/bin/python3").is_file()):
            raise RuntimeError("Pinned control archive has no portable Python")
        shutil.copytree(ROOT / "terminal", bundle / "terminal",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        if row["platform"] == "macos-arm64":
            tools = bundle / "terminal" / "metal" / "tools"
            tools.mkdir(parents=True)
            for name in ("metal_repack.py", "stream_experts.py"):
                shutil.copy2(ROOT / "engine" / "metal" / "tools" / name, tools / name)
        for name in ("aurapro_engine.py", "desktop_runtime.py", "LICENSE", "UPSTREAM.json"):
            shutil.copy2(ROOT / name, bundle / name)
        (bundle / "BUNDLE.json").write_text(
            json.dumps({"version": VERSION, "platform": row["platform"]}), encoding="utf-8"
        )
        (bundle / "ENGINE-ASSETS.json").write_text(json.dumps([
            {**entry, "url": name} for name, entry in engines.items() if entry["platform"] == row["platform"]
        ]), encoding="utf-8")
        dest = output / f"aurapro-engine-control-{VERSION}-{row['platform']}.{row['format']}"
        if row["format"] == "zip":
            with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as tar:
                for file in bundle.rglob("*"):
                    if file.is_file():
                        tar.write(file, file.relative_to(bundle).as_posix())
        else:
            with tarfile.open(dest, "w:gz") as tar:
                for file in bundle.iterdir():
                    tar.add(file, arcname=file.name)
        return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("matrix", "control", "native", "verify"))
    parser.add_argument("item", nargs="?")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    controls, engines = manifest()
    if args.kind == "matrix":
        print(json.dumps(list(engines)))
        return
    if args.kind == "verify":
        published = json.load(sys.stdin)
        actual = {row["name"]: row for row in published["assets"] if row["state"] == "uploaded"}
        for name, entry in engines.items():
            row = actual[name]
            if row["digest"] != "sha256:" + entry["sha256"] or row["size"] != entry["size"]:
                raise RuntimeError(f"{name}: published native digest mismatch")
        for platform, entry in controls.items():
            name = f"aurapro-engine-control-{VERSION}-{platform}.{entry['format']}"
            if not actual[name].get("digest", "").startswith("sha256:"):
                raise RuntimeError(f"{name}: no published control digest")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if args.kind == "control":
        print(control(controls[args.item], engines, args.out))
    else:
        target = args.out / args.item
        download(engines[args.item], target)
        print(target)


if __name__ == "__main__":
    main()
