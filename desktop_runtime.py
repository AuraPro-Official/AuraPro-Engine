"""Private desktop entrypoint for the AuraPro-Engine control bundle."""

import json
import os
import sys
from pathlib import Path

from aurapro_engine import VERSION, managed_prepare


def main(args):
    root = Path(__file__).resolve().parent
    action, *options = args
    os.environ["STRATA_COMPONENTS_DIR"] = str(root / "components")
    os.environ["STRATA_REQUIRE_SIGNED_COMPONENTS"] = "1"
    if action == "serve":
        from serve import server

        os.environ["STRATA_NO_ACQUIRE"] = "1"
        return server.main([*options, "--offline"])
    if action == "model":
        return managed_prepare(root, options)
    from strata_control import desktop_release

    if action == "select" and len(options) == 1:
        package, _ = desktop_release.selection(root, options[0])
        result = {"name": package.url, "sha256": package.sha256, "size": package.size}
    elif action == "install" and len(options) == 2:
        result = desktop_release.install(root, options[0], Path(options[1]))
    elif action == "version" and not options:
        result = {"version": VERSION}
    else:
        raise ValueError("Unknown AuraPro Engine desktop command")
    print("AURAPRO_RESULT:" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
