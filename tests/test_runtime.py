import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import Request

import aurapro_engine as engine
import setup
from serve import server
from strata_control import mac_fit, pro_download
from tools import mtp_fetch


class DownloadPolicyTests(unittest.TestCase):
    def test_all_presets_use_allowed_pinned_files(self):
        self.assertEqual(list(engine.MODELS), ["Q2_0", "IQ2_XS", "Q2_K_XL", "IQ3_XXS", "Q3_K_XL", "IQ4_XS", "Q4_K_XL"])
        for model, (family, quant) in engine.MODELS.items():
            with self.subTest(model=model):
                fam = setup.FAMILIES[family]
                urls = [
                    setup.model_hf(fam, quant) + setup.model_file(fam, quant, i)
                    for i in range(1, setup.model_shards(fam, quant) + 1)
                ]
                urls.append(fam["mmproj_hf"] + fam["mmproj"])
                for url in urls:
                    pro_download.check_url(url)
                    size, sha256 = setup.file_pin(url)
                    self.assertGreater(size, 0)
                    self.assertRegex(sha256, r"^[a-f0-9]{64}$")

    def test_block_unapproved_downloads(self):
        lite = setup.OWN_PATH_FAMILIES["qwen36"]["hf"]
        flash = setup.FAMILIES["unsloth"]
        urls = [
            setup.model_hf(flash, "UD-IQ1_M") + setup.model_file(flash, "UD-IQ1_M", 1),
            setup.model_hf(flash, "UD-Q4_K_XL") + setup.model_file(flash, "UD-Q4_K_XL", 1),
            lite + "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
            lite + "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
            lite + "Qwen3.6-35B-A3B-Q8_0.gguf",
            lite + "mmproj-F16.gguf",
            lite.replace("huggingface.co", "modelscope.cn") + "mmproj-F16.gguf",
            lite.replace("huggingface.co", "huggingface.co.example.test") + "mmproj-F16.gguf",
            lite.replace(setup.HF_REVISIONS["unsloth/Qwen3.6-35B-A3B-MTP-GGUF"], "main") + "mmproj-F16.gguf",
            "http://cdn-lfs.hf.co/object",
            "https://cdn-lfs.hf.co:81/object",
        ]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(PermissionError):
                pro_download.check_url(url)

    def test_q4_v31_uses_only_revised_shards_and_a_separate_profile(self):
        family, quant = engine.MODELS["Q4_K_XL"]
        self.assertEqual(quant, setup.UNSLOTH_V31_Q4_K_XL)
        self.assertIn(quant, mac_fit.app_sizes(family))
        fam = setup.FAMILIES[family]
        base = setup.model_hf(fam, quant)
        self.assertIn(f"/{setup.UNSLOTH_V31_REVISION}/UD-Q4_K_XL/", base)
        self.assertEqual(setup.model_shards(fam, quant), 4)
        for index in range(1, 5):
            name = setup.model_file(fam, quant, index)
            self.assertEqual(name, setup.model_file(fam, "UD-Q4_K_XL", index))
            pin = setup.UNSLOTH_Q4_V31_SHARDS[name]
            self.assertEqual(setup.file_pin(base + name), pin)
            self.assertNotEqual(setup.file_pin(setup.model_hf(fam, "UD-Q4_K_XL") + name), pin)
            with patch.object(setup, "fam_pin") as legacy_pin:
                self.assertEqual(setup.model_shard_pin(fam, quant, Path(name)), pin)
                legacy_pin.assert_not_called()
        self.assertEqual(engine.profile_path(Path("data"), "Q4_K_XL").name, "strata-unsloth-ud-q4_k_xl-v31.json")

    def test_redirect_checks_destination_and_preserves_head(self):
        fam = setup.FAMILIES["unsloth"]
        url = fam["mmproj_hf"] + fam["mmproj"]
        pro_download.check_url(url)
        redirect = pro_download.HubRedirect()
        req = Request(url, method="HEAD")
        cdn = "https://cas-bridge.xethub.hf.co/xet/object?signature=test"
        self.assertEqual(redirect.redirect_request(req, None, 302, "", {}, cdn).get_method(), "HEAD")
        with self.assertRaises(PermissionError):
            redirect.redirect_request(req, None, 302, "", {}, "https://example.test/file")

    def test_mtp_is_fixed_and_allows_hub_cache_redirect(self):
        expected = pro_download.MTP_REVISION
        self.assertIn(f"/resolve/{expected}/", mtp_fetch.REPO)
        path = f"/api/resolve-cache/models/{pro_download.MTP_REPO}/{expected}/model.safetensors.index.json"
        pro_download.check_url("https://huggingface.co" + path + "?download=true")
        with self.assertRaises(PermissionError):
            pro_download.check_url("https://huggingface.co" + path.replace(expected, "0" * 40))

    def test_unapproved_url_never_reaches_upstream_transfer(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "_open_net") as opened:
            dest = Path(tmp) / "not-created" / "model.gguf"
            with self.assertRaises(ValueError):
                setup.download("https://example.test/model.gguf", dest)
            opened.assert_not_called()
            self.assertFalse(dest.parent.exists())


class RuntimeTests(unittest.TestCase):
    def test_model_disk_space_reports_sub_gigabyte_deficit_in_bytes(self):
        total, free = 22_853_663_008, 22_710_263_808
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            setup.shutil, "disk_usage", return_value=SimpleNamespace(free=free)
        ), patch.object(setup, "fail", side_effect=SystemExit(1)) as fail:
            target = Path(tmp) / "lite_Q4.gguf"
            with self.assertRaises(SystemExit):
                setup.require_model_disk_space(target, total)
            self.assertIn("need 22.854 GB, 22.710 GB free (short by 144 MB)", fail.call_args.args[0])
            self.assertEqual(fail.call_args.kwargs, {
                "kind": "no_space", "path": tmp, "free_bytes": free, "needed_bytes": total,
            })

    def test_model_disk_space_credits_resumable_bytes_and_keeps_partial(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            setup.shutil, "disk_usage", return_value=SimpleNamespace(free=4)
        ), patch.object(setup, "warn"), patch.object(setup, "fail") as fail:
            target = Path(tmp) / "model.gguf"
            partial = target.with_name("model.gguf.part")
            partial.write_bytes(b"GGUF")
            setup.require_model_disk_space(target, 8)
            fail.assert_not_called()
            self.assertEqual(partial.read_bytes(), b"GGUF")

    def test_default_data_dir_preserves_early_installations(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(Path, "home", return_value=Path(tmp)):
            home = Path(tmp)
            current = home / ".aurapro-engine"
            legacy = home / ".aurapro-pro"
            self.assertEqual(engine.parse_args(["serve"]).data_dir, current)
            legacy.mkdir()
            self.assertEqual(engine.parse_args(["serve"]).data_dir, legacy)
            current.mkdir()
            self.assertEqual(engine.parse_args(["serve"]).data_dir, current)
            custom = home / "custom"
            self.assertEqual(engine.parse_args(["--data-dir", str(custom), "serve"]).data_dir, custom)

    def test_environment_isolated_and_offline_at_start(self):
        with patch.dict(os.environ, {"Strata_Source": "other", "HF_ENDPOINT": "other", "STRATA_API_KEY": "other"}):
            engine.configure_environment(Path("local"), prepare=False, backend="auto")
            self.assertNotIn("HF_ENDPOINT", os.environ)
            self.assertNotIn("STRATA_API_KEY", os.environ)
            self.assertNotIn("Strata_Source", os.environ)
            self.assertEqual(os.environ["STRATA_NO_ACQUIRE"], "1")

    def test_prepare_is_noninteractive_and_does_not_start(self):
        args = engine.parse_args(["prepare", "--model", "Q2_0"])
        flags = engine.prepare_args(Path("local"), args)
        self.assertIn("--no-start", flags)
        self.assertIn("--no-browser", flags)
        self.assertEqual(flags[flags.index("--kv") + 1], "int8")
        self.assertEqual(flags[flags.index("--family") + 1], "unsloth")
        self.assertEqual(flags[flags.index("--model") + 1], "UD-Q2_0")

    def test_retired_models_can_only_serve_local_profiles(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, "main", return_value=0) as main:
            data = Path(tmp)
            for model, family, quant in (
                ("IQ1_M", "unsloth", "ud-iq1_m"),
                ("lite_Q4", "qwen36", "ud-q4_k_xl"),
                ("lite_Q6", "qwen36", "ud-q6_k_xl"),
            ):
                with self.subTest(model=model), patch("sys.stderr"):
                    self.assertNotIn(model, engine.MODELS)
                    with self.assertRaises(SystemExit):
                        engine.parse_args(["prepare", "--model", model])
                    args = engine.parse_args(["serve", "--model", model])
                    config = engine.profile_path(data, args.model)
                    self.assertEqual(config.name, f"strata-{family}-{quant}.json")
                    config.parent.mkdir(exist_ok=True)
                    config.write_text("{}", encoding="utf-8")
                    self.assertEqual(engine.serve_model(data, args), 0)
                    self.assertIn(str(config), main.call_args.args[0])
                    self.assertIn("--offline", main.call_args.args[0])

    def test_prepare_does_not_use_another_installation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "other_installs") as discover:
            data = Path(tmp).resolve() / "engine-data"
            args = engine.parse_args(["--data-dir", str(data), "prepare"])
            self.assertEqual(setup.data_folder_for(args), (data, []))
            self.assertTrue(data.is_dir())
            discover.assert_not_called()

    def test_start_uses_only_the_selected_local_profile(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(server, "main", return_value=0) as main:
            data = Path(tmp)
            config = engine.profile_path(data, "Q2_0")
            config.parent.mkdir()
            config.write_text(json.dumps({"exe": "component:strata-engine"}), encoding="utf-8")
            self.assertEqual(engine.serve_model(data, engine.parse_args(["serve"])), 0)
            args, = main.call_args.args
            self.assertEqual(args, ["--engine", "strata", "--config", str(config), "--port", "18882", "--offline"])

    def test_images_cannot_read_a_url_or_disk(self):
        self.assertEqual(server.Vision.load("data:image/png;base64,YWJj"), b"abc")
        for source in ("C:/secret.png", "file:///home/local/secret.png", "https://example.test/image.png"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                server.Vision.load(source)

    def test_desktop_and_agent_components_are_not_imported(self):
        for path in ("app", "terminal/strata_cli", "terminal/serve/web", "terminal/serve/mcp.py", "terminal/tools/quantscope.py"):
            self.assertFalse((engine.ROOT / path).exists(), path)


class MockEngineTests(unittest.TestCase):
    def test_serve_on_loopback_without_exposing_upstream_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready.json"
            code = (
                "import aurapro_engine as p, sys; from pathlib import Path; "
                "p.configure_environment(Path(sys.argv[1]), prepare=False, backend='auto'); "
                "from serve import server; "
                "raise SystemExit(server.main(['--engine','mock','--host','0.0.0.0',"
                "'--port','0','--ready-file',sys.argv[2]]))"
            )
            log_path = Path(tmp) / "server.log"
            with log_path.open("w+", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-B", "-s", "-c", code, tmp, str(ready)],
                    cwd=engine.ROOT, stdout=log, stderr=log,
                )
                try:
                    deadline = time.monotonic() + 15
                    data = {}
                    while not data and proc.poll() is None and time.monotonic() < deadline:
                        try:
                            data = json.loads(ready.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            time.sleep(0.05)
                    if not data:
                        log.seek(0)
                        self.fail(f"Mock server did not become ready: {log.read()}")
                    self.assertEqual(data["bound_host"], "127.0.0.1")
                    port = data["bound_port"]

                    def request(method, path, payload=None, headers=None):
                        conn = HTTPConnection("127.0.0.1", port, timeout=5)
                        try:
                            conn.request(method, path, payload, headers or {})
                            response = conn.getresponse()
                            return response.status, json.loads(response.read())
                        finally:
                            conn.close()

                    self.assertEqual(request("GET", "/health")[0], 200)
                    status, models = request("GET", "/v1/models")
                    self.assertEqual(status, 200)
                    payload = json.dumps({"model": models["data"][0]["id"], "messages": [{"role": "user", "content": "Hello"}]})
                    status, response = request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
                    self.assertEqual(status, 200)
                    self.assertTrue(response["choices"])
                    for method, route in (("GET", "/"), ("GET", "/mcp"), ("POST", "/v1/responses"), ("OPTIONS", "/v1/chat/completions")):
                        self.assertEqual(request(method, route)[0], 404)
                    self.assertEqual(request("GET", "/health", headers={"Host": "foreign.example"})[0], 403)
                    self.assertEqual(request("GET", "/health", headers={"Origin": "https://foreign.example"})[0], 403)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                    proc.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
