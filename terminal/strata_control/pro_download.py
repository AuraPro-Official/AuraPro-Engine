"""The download origins allowed during AuraPro Engine model preparation."""

import re
import urllib.request
from urllib.parse import urlsplit

from strata_control import acquisition, netpolicy
from strata_control.pro_catalog import MODELS


MODEL_REVISIONS = {
    "unsloth/Qwen3.8-Flash-Next-GGUF": frozenset({
        "d96201194f103c906994f283f504209dc8d2a10c",
    }),
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": frozenset({
        "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
    }),
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": frozenset({
        "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    }),
}
MTP_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MTP_REPO = "Qwen/Qwen3.8-Flash-Next"
HUB = "https://huggingface.co"
_FILE = re.compile(r"/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/((?:[A-Za-z0-9_-]+/)?[A-Za-z0-9_.-]+)")
_CACHE = re.compile(r"/api/resolve-cache/models/([^/]+/[^/]+)/([0-9a-f]{40})/([A-Za-z0-9_.-]+)")
_MTP_FILE = re.compile(r"model(?:\.safetensors\.index\.json|-[0-9]+-of-[0-9]+\.safetensors)")
_FLASH_QUANTS = "|".join(re.escape(quant.removesuffix("-v31")) for family, quant in MODELS.values() if family == "unsloth")
_FLASH_SHARD = re.compile(rf"({_FLASH_QUANTS})/Qwen3\.8-Flash-Next-\1-[0-9]{{5}}-of-[0-9]{{5}}\.gguf")
_LITE_FILES = {f"Qwen3.6-35B-A3B-{quant}.gguf" for family, quant in MODELS.values() if family == "qwen36"}


def allowed_model_ref(repo: str, revision: str) -> bool:
    return revision in MODEL_REVISIONS.get(repo, ())


def allowed_model_path(repo: str, revision: str, path: str) -> bool:
    if not allowed_model_ref(repo, revision):
        return False
    if repo == "unsloth/Qwen3.8-Flash-Next-GGUF":
        return bool(_FLASH_SHARD.fullmatch(path))
    if repo == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF":
        return path in _LITE_FILES or path == "mmproj-F16.gguf"
    return path == "mmproj-Qwen3.8-Flash-Next-BF16.gguf"


def check_url(url: str) -> None:
    target = urlsplit(url)
    host = target.hostname or ""
    if (
        target.scheme != "https" or target.port not in (None, 443)
        or target.username is not None or target.password is not None or target.fragment
    ):
        raise PermissionError("Model downloads require an approved HTTPS origin")
    if host == "huggingface.co":
        cache = _CACHE.fullmatch(target.path)
        match = _FILE.fullmatch(target.path) or cache
        if match and (not target.query or cache):
            repo, revision, filename = match.groups()
            if allowed_model_path(repo, revision, filename):
                return
            if repo == MTP_REPO and revision == MTP_REVISION and _MTP_FILE.fullmatch(filename):
                return
    elif host.endswith((".hf.co", ".huggingface.co")):
        return  # Hub's signed storage redirects; the caller still verifies the bytes.
    raise PermissionError(f"Unapproved model download origin or file: {host}")


class HubRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and req.get_method() == "HEAD":
            redirected.method = "HEAD"
        return redirected


def open_model_url(request, timeout=None):
    acquisition.require("an AuraPro Engine model download")
    check_url(getattr(request, "full_url", request))
    return netpolicy.build_opener(HubRedirect()).open(request, timeout=timeout)
