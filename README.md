# AuraPro Engine

`AuraPro-Engine` is intended to be AuraPro's single local inference backend. The
initial implementation is a source fork of Strata, not a
repackaged copy of the upstream desktop application. The exact upstream release
and commit are recorded in [UPSTREAM.json](UPSTREAM.json). The upstream MIT
copyright and third-party notices are retained.

## Runtime Roadmap

The goal is for AuraPro Engine to replace both the desktop app's standard
llama.cpp runtime and its Pro runtime. It starts with the Pro model presets.
Before switching the desktop app, the unified engine needs to support the
standard models and platforms that AuraPro intends to keep. The end state is
one inference backend, not two engines behind a shared interface.

## Release 0.0.3

Restores a missing preparation helper so downloaded Pro models can be
prepared without downloading the model files again.

The desktop Pro installer on Windows and Linux x64 uses the AuraPro Engine
control package. It downloads one compatible CUDA or HIP component from the
same release. Apple Silicon builds are also packaged for standalone testing;
the desktop's macOS Pro installation stays on its existing path in this release.
Standard inference is unchanged on all platforms.

The control package runs the AuraPro-maintained code in this repository. The
first release redistributes unmodified, signed Strata v0.2.20 native binaries
and its pinned portable Python environment; it does not claim independent
native builds. The packaging job checks their pinned hashes. Existing upstream
component IDs, executable names and notices are kept for compatibility and
attribution.

## Scope

- The native engine and its CMake projects are in `engine/`. The managed runtime
  supports CUDA and HIP on Windows/Linux x64 and Metal on Apple Silicon.
- The only public runtime commands are `models`, `prepare`, and `serve`.
  There is no upstream desktop UI, terminal launcher, coding agent, cloud model
  provider, tunnel, or updater in the AuraPro Engine runtime interface.
- Preparation uses pinned Hugging Face model files and their hashes. It does not
  load custom URLs, mirror settings, or model lists from a release feed.
- Serving binds to loopback, does not download files, and exposes only chat,
  model discovery, health, and context/status endpoints. Image inputs must be
  embedded data URLs. Offline mode is a runtime policy, not an OS sandbox.

## Models

- Unsloth Qwen3.8-Flash-Next: `Q2_0` (default), `IQ2_XS`, `Q2_K_XL`,
  `IQ3_XXS`, `Q3_K_XL`, `IQ4_XS`, `Q4_K_XL` (the pinned Unsloth v3.1 layout).
- Retired `IQ1_M`, `lite_Q4` and `lite_Q6` profiles can still run offline if
  already prepared, but cannot be downloaded again. The Lite profiles remain
  CUDA/HIP-only in AuraPro Engine.

Model weights are not included. Model licenses are separate from this
project's source license.

## Develop

Use a Python 3.12 virtual environment with the pinned
`terminal/requirements.txt`. Install dependencies explicitly; the runtime
never runs an environment installer.

```sh
python -m pip install -r terminal/requirements.txt
python aurapro_engine.py models
python -m unittest discover -s tests -v
```

Build from `engine/` with the appropriate upstream CMake preset or a matching
local toolchain. For a CPU-only source check:

```sh
cd engine
cmake --preset cpu-check
cmake --build --preset cpu-check
ctest --preset cpu-check
```

The CUDA/HIP/Metal developer packaging scripts are under `tools/packaging/`.
They are separate from the end-user runtime. A local engine package has a
manifest with hashes, and is installed by placing the package directory under
`<data-dir>/components/engine/`. Use only engines you built or obtained from
an independently trusted source; hashes in a local manifest do not verify the
publisher. With a compatible engine present:

```sh
python aurapro_engine.py --data-dir ./engine-data prepare --model Q2_0
python aurapro_engine.py --data-dir ./engine-data serve --model Q2_0
```

The OpenAI-compatible base URL is `http://127.0.0.1:18882/v1`. Data defaults to
`~/.aurapro-engine`. If that path does not exist and `~/.aurapro-pro` already
exists as a directory, the CLI reuses it without moving any files. Use
`--data-dir` to select a different directory. This standalone project does not
change the regular desktop inference path or replace an installed runtime by
itself.
