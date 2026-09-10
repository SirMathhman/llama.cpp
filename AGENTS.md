# AGENTS.md

Guidance for AI coding agents working in this repository. This file is referenced by
[CONTRIBUTING.md](CONTRIBUTING.md), the [PR template](.github/pull_request_template.md), and the
repo skills. Keep it concise and actionable.

## What this project is

`llama.cpp` is a plain C/C++ LLM/VLM inference engine built on the [ggml](https://github.com/ggml-org/ggml)
tensor library. It targets a wide range of hardware (CPU, CUDA, Metal, Vulkan, SYCL, HIP, OpenCL, ...)
with minimal dependencies.

## Layered architecture

Strict dependency direction — top depends on bottom, never the reverse:

```
tools/   CLI apps (cli, server, bench, imatrix, quantize, ...)
   │
common/  shared C++ utilities (sampling, chat templates, args, logging)
   │
src/     the llama library (model loading, graph building, KV cache, samplers)
   │
ggml/    tensor library + pluggable compute backends
```

- **`ggml`** = low-level tensor ops + backends. Public API: `ggml/include/ggml.h`.
- **`llama`** = model inference built on ggml. Public C API: `include/llama.h`.
- Backends depend only on `ggml/src/ggml-backend-impl.h`, never on `src/` or `common/`.

Key files to internalize:

- `ggml/include/ggml.h` — tensor API + `enum ggml_op` (the op catalog).
- `ggml/src/ggml-backend-impl.h` — the backend vtable (interface) pattern.
- `ggml/src/ggml-cuda/ggml-cuda.cu` — a complete backend example.
- `src/llama-graph.cpp` — how the model forward-pass graph is assembled.

## Build & test

Build system is **CMake only** (the `Makefile` is a deprecated stub). See [docs/build.md](docs/build.md).

```sh
# Build (CPU)
cmake -B build
cmake --build build --config Release -j 8

# Test (CTest, label "main")
cd build && ctest -L main --verbose --timeout 900

# Single test
ctest -R test-name --verbose
```

- GPU toggles are `GGML_*`, **not** `LLAMA_*`: `GGML_CUDA`, `GGML_METAL`, `GGML_VULKAN`, `GGML_SYCL`, `GGML_HIP`, `GGML_OPENCL`.
- `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_SERVER`, `LLAMA_FATAL_WARNINGS` are the main top-level options.
- Windows CI uses the **LLVM/clang** toolchain (`cmake/x64-windows-llvm.cmake`) with `Ninja Multi-Config`, not plain MSVC.
- Python tooling (active in CI): `flake8 --plugins=flake8-no-print` (lint) and `ty check` (type-check). No ruff/black.

## Code conventions

Standards: **C11** and **C++17** (see `ggml/CMakeLists.txt`). Style is enforced by [`.clang-format`](.clang-format)
(4-space indent, 120-col, K&R braces, left-aligned pointers) and [`.clang-tidy`](.clang-tidy).

- **ASCII only** in code and comments — no em-dashes, unicode arrows, `…`, `×`.
- **Naming:** `snake_case` for functions/vars/types; optimize for the **longest common prefix**
  (`number_small`, not `small_number`). Constants/enums `UPPER_SNAKE_CASE` with a module prefix
  (`GGML_OP_ADD`, `LLAMA_TOKEN_NULL`). C/C++ filenames lowercase-with-dashes.
- **Public API:** sized ints (`int32_t`) and `size_t` for sizes/offsets. `struct foo {}`, not
  `typedef struct foo {} foo`. Opaque types get a `_t` suffix.
- **Keep it simple:** avoid third-party deps, extra files/headers, and fancy STL/templates. Prefer basic
  `for` loops. A simpler change doing 90% is preferred over a complex one doing 100%.
- **Tensors are row-major:** dim 0 = columns, 1 = rows, 2 = matrices.
- **Non-standard matmul:** `ggml_mul_mat(ctx, A, B)` computes `C = B·Aᵀ` (i.e. `Cᵀ = A·Bᵀ`). This trips up
  graph authors constantly — double-check orientation.

## ggml / backend gotchas

- **Do not mutate the cgraph** as a shortcut inside a backend — it's an unresolved architectural question.
- **`supports_op` gating must be scoped exactly** — a condition meant for a few quant types must not
  silently enable/disable everything else.
- **No hardcoded warp/lane size** — use `ggml_cuda_get_physical_warp_size()` (32 CUDA, 64 HIP).
- **RoPE:** never hand-roll sin/cos — use `ggml_rope_ext`. If it can't express the need, open an issue;
  don't PR a custom implementation.
- **QKV:** split the _activation_ with `ggml_view`, not the _weight_ tensor; rely on ggml broadcasting.
- **New/changed ggml op** → update [docs/ops.md](docs/ops.md) + `docs/ops/<backend>.csv` and add
  `test-backend-ops` cases (needs ≥2 backends).
- **Hoist per-layer-invariant tensors** out of the layer loop to avoid `GGML_SCHED_MAX_SPLIT_INPUTS` splits.

## Adding a new model

Follow [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) and the
[`add-new-model` skill](skills/add-new-model/SKILL.md). Highlights:

- **CPU-only first**; other backends land as follow-up PRs.
- Don't validate the same hparam in both Python conversion and C++ load — pick one owner.
- Load-bearing hparams must **hard-error** if missing; only genuinely-optional ones get a fallback.
- Don't bake a default chat template into C++ — inject it at conversion time.
- Do constant tensor modifications (permutes, `norm(1+weight)`) **at conversion time**, not in the graph.
- Gate on **hparam/capability**, not the `model.arch` enum. New tensor names go through `tensor_mapping.py`.
- Test the **quantized-KV path** (`-ctk/-ctv q8_0`), not just f16.

## Security (blocking)

GGUF metadata, tensor shapes, tokenizer/grammar input, and all server/RPC fields are **attacker-controlled** —
bound them before use. See the [`code-review` skill](skills/code-review/SKILL.md) for the full checklist.

- Validate `ne[i]*nb[i]` / nbytes **before** arithmetic (overflow → undersized alloc → heap overflow).
- Overflow checks must run **before** padding/alignment macros (they wrap to 0 near `SIZE_MAX`).
- Cap GGUF string/array lengths before sizing loops/buffers; check element **type** before casting
  `gguf_get_arr_data()` to `float*`/`int32_t*`.
- `GGML_ASSERT` on file-derived values **aborts** — throw instead where the caller catches.
- Clamp client-supplied server JSON ints to non-negative + upper bound before index/pointer math.

## AI usage policy

From [CONTRIBUTING.md](CONTRIBUTING.md) — these are **mandatory**:

- AI-generated code is allowed, but the contributor is **100% responsible** for every line.
- **Disclose** AI usage; undisclosed use may result in a ban.
- **Never** use AI to write PR descriptions, commit messages, issues, or replies to humans.
- Use `Assisted-by:` (never `Co-authored-by:`) for AI contributions.
- Features must start as an **issue**, not a PR. Bug fixes need a reproducible issue + a **regression test**
  that fails before / passes after.
- Run the full CI locally ([ci/README.md](ci/README.md)); verify perplexity (`llama-perplexity`) and perf
  (`llama-bench`).

## Documentation index

Link, don't duplicate. Key docs:

- [docs/build.md](docs/build.md) — build guide for every backend.
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding a model arch.
- [docs/development/debugging-tests.md](docs/development/debugging-tests.md) — running/debugging a single test.
- [docs/ops.md](docs/ops.md) — ggml op × backend support matrix.
- [docs/multi-gpu.md](docs/multi-gpu.md), [docs/speculative.md](docs/speculative.md), [docs/llguidance.md](docs/llguidance.md).
- [ci/README.md](ci/README.md) — running full CI locally.
- [CONTRIBUTING.md](CONTRIBUTING.md) — conventions and PR process.
