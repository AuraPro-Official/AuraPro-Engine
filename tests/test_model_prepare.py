import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import aurapro_engine as engine
import desktop_runtime
import setup
from strata_control import engine_component, pro_download
from tools import mtp_rt
from tools.gguf_writer import GGUFWriter


def write_shards(folder):
    paths = [folder / f"model-{i:05d}-of-00004.gguf" for i in range(1, 5)]
    for i, path in enumerate(paths):
        writer = GGUFWriter()
        writer.add("split.count", len(paths), "u16")
        writer.add("split.no", i, "u16")
        writer.add("split.tensors.count", 5)
        if i == 0:
            writer.add("general.architecture", "qwen4exp")
            writer.add("tokenizer.ggml.model", "gpt2")
            writer.add("tokenizer.ggml.tokens", ["a"])
            writer.add("tokenizer.ggml.merges", [], "array:string")
            writer.add("tokenizer.chat_template", "{{ messages[0]['content'] }}")
            writer.add_f32("blk.0.ffn_gate_inp.weight", np.full((2, 2), 0.1))
            writer.add_f32("per_layer_token_embd.weight", np.ones((2, 2)))
        else:
            role = ("gate", "up", "down")[i - 1]
            writer.add_q2_0(f"blk.0.ffn_{role}_exps.weight", np.ones((2, 64)), [64, 1, 2])
        writer.write(path)
    return paths


class ModelPrepareTests(unittest.TestCase):
    def run_tool(self, script, *args):
        command = [sys.executable, str(setup.ROOT / "tools" / script), *args]
        env = {"STRATA_NO_ACQUIRE": "0", "STRATA_EXIT_WITH_PARENT": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        with patch.object(setup, "CONTROL", True), patch.dict(os.environ, env):
            return setup.run(command, env=dict(os.environ, **setup.gguf_py_env()), check=False, quiet=True)

    def run_packer(self, model, source, output, *flags):
        family, _ = engine.MODELS[model]
        return self.run_tool("iq_pack.py", "--gguf", str(source), "--out", str(output), *flags,
                             *setup.FAMILIES[family].get("pack_args", []))

    def test_v31_prepares_split_weights_and_tokenizer(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = write_shards(folder)[0]
            output = folder / "pack"
            result = self.run_packer("IQ4_XS", source, output)
            self.assertEqual(result.returncode, 0, result.stdout)
            for filename in ("dense.bin", "index.txt", "native_experts.txt", "tokenizer/vocab.json",
                             "tokenizer/chat_template.jinja"):
                self.assertGreater((output / filename).stat().st_size, 0, filename)
            self.assertFalse((output / "experts.bin").exists())
            self.assertIn("round-to-nearest-even", (output / "conversions.json").read_text(encoding="utf-8"))

            for _ in range(2):
                result = self.run_packer("Q4_K_XL", source, output, "--experts-bin")
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual((output / "experts.bin").stat().st_size, 3 * 2 * 18)

    def test_missing_shard_does_not_publish_pack(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = write_shards(folder)
            paths[-1].unlink()
            output = folder / "pack"
            result = self.run_packer("IQ4_XS", paths[0], output)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(paths[-1].name, result.stdout)
            self.assertNotIn("ModuleNotFoundError", result.stdout)
            self.assertFalse((output / "native_experts.txt").exists())

    def test_mtp_prepares_one_expert_from_local_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            tensors = {
                "mtp.layers.0.mlp.experts.gate_up_proj": (1, 2 * mtp_rt.FF, mtp_rt.H),
                "mtp.layers.0.mlp.experts.down_proj": (1, mtp_rt.H, mtp_rt.FF),
                "mtp.layers.0.norm.weight": (2,),
            }
            manifest = []
            for name, shape in tensors.items():
                content = np.full(shape, 0x3F80, dtype="<u2").tobytes()
                file = folder / f"{name}.bin"
                file.write_bytes(content)
                manifest.append({"name": name, "shape": shape, "file": file.name,
                                 "sha256": hashlib.sha256(content).hexdigest()})
            (folder / "mtp-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            draft = folder / "mtp-q2_0.gguf"
            result = self.run_tool("mtp_pack.py", "--src", str(folder), "--experts", "q2_0", "--out", str(draft))
            self.assertEqual(result.returncode, 0, result.stdout)

            output = folder / "rt"
            with patch.object(mtp_rt, "NE", 1), patch.object(
                sys, "argv", ["mtp_rt.py", "--gguf", str(draft), "--out", str(output)]
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(mtp_rt.main(), 0)
            self.assertEqual((output / "experts.bin").stat().st_size, mtp_rt.BLOB)
            self.assertEqual((output / "dense.bin").stat().st_size, 2 * 4)

    @unittest.skipIf(sys.platform == "darwin", "CUDA/HIP model preparation path")
    def test_desktop_prepare_writes_profile_and_reuses_downloaded_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            data = folder / "data"
            profiles = folder / "release"
            profiles.mkdir()
            shards = write_shards(folder)
            family, quant = engine.MODELS["IQ4_XS"]
            fam = setup.FAMILIES[family]
            self.assertEqual(setup.model_shards(fam, quant), len(shards))
            fixtures = {setup.model_file(fam, quant, i): shard for i, shard in enumerate(shards, 1)}
            # Exercise the projector path without running a native encoder.
            fixtures[fam["mmproj"]] = shards[0]
            (folder / setup.VEXE).touch()
            gpu = {"index": 0, "name": "Test GPU", "vram_gb": 24, "arch": "120", "driver": "621.14"}
            native = SimpleNamespace(root=folder, version="0.1.66", platform="test", backend="cuda")
            transfers = []

            def local_download(url, target, *_):
                pro_download.check_url(url)
                if target.is_file() and setup.done(target):
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(fixtures[target.name], target)
                setup.mark(target)
                transfers.append(target.name)

            def fixture_pin(_fam, _model, path, **_):
                content = fixtures[path.name].read_bytes()
                return len(content), hashlib.sha256(content).hexdigest()

            flags = ["model", "--setup", "--yes", "--no-start", "--no-browser", "--backend", "cuda",
                     "--family", family, "--model", quant, "--context", "16384", "--kv", "int8",
                     "--drafter", "mtp", "--vision", "yes", "--data-dir", str(data),
                     "--models-dir", str(data / "models")]
            hardware = {"gpus": lambda: [gpu.copy()], "amd_gpus": lambda: [],
                        "ram_gb": lambda: 128, "cpu_info": lambda: ("Test CPU", True, True),
                        "page_file_gb": lambda: None, "pcie_link": lambda _: None, "free_gb": lambda _: 256}
            with (
                patch.dict(os.environ, {"PYTHONDONTWRITEBYTECODE": "1"}),
                patch.object(desktop_runtime, "__file__", str(profiles / "desktop_runtime.py")),
                patch.object(engine_component, "resolve", return_value=native),
                patch.multiple(setup, CONTROL=True, _CONTROL_HOME=profiles, GPU_PICK=None, OLD_GPUS=None, **hardware),
                patch.dict(setup.SHAPE_ENGINE, clear=True),
                patch.object(setup, "write_run_script"),
                patch.object(setup, "fetch_mtp", return_value=False) as fetch_mtp,
                patch.object(setup, "model_shard_pin", side_effect=fixture_pin),
                patch.object(setup, "download", side_effect=local_download),
                redirect_stdout(io.StringIO()) as log,
            ):
                self.assertEqual(desktop_runtime.main(flags), 0, log.getvalue())
                self.assertCountEqual(transfers, fixtures)
                fetch_mtp.assert_called_once()
                profile = profiles / f"strata-{fam['tag'].lower()}{quant.lower()}.json"
                config = json.loads(profile.read_text(encoding="utf-8"))
                self.assertEqual(config["exe"], "component:strata-engine")
                self.assertEqual(Path(config["vision"]["mmproj"]).name, fam["mmproj"])
                pack = Path(config["args"][config["args"].index("--pack") + 1])
                self.assertTrue((pack / "native_experts.txt").is_file())
                with patch.object(setup, "run", side_effect=AssertionError("unexpected re-pack")):
                    self.assertEqual(desktop_runtime.main(flags), 0, log.getvalue())
                self.assertCountEqual(transfers, fixtures)
