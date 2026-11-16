# Building this source integration

This publication distributes source, not a prebuilt wheel or container.
The original Backport README's Docker Hub tags identify the author's builds;
they do not contain this repository's rebased integration.

## Environment and scope

The privately tested environment is recorded in [validation.md](validation.md).
Use an isolated Linux x86_64 builder with a CUDA development toolkit, GCC 13+,
Python 3.12, the repository's required Torch version, Rust, and enough disk/RAM.
Building only SM80 avoids compiling unneeded GPU architectures. The inherited
Dockerfile is source material, not a separately validated public release recipe.

Use uv to create a virtual environment and install the matching build/runtime
dependencies from the repository requirements. CUDA FetchContent and Cargo
dependencies also need to be available. Follow the inherited
[source-build instructions](../getting_started/installation/gpu.md)
alongside the pinned versions in this source tree. Do not mix a different vLLM
wheel's native extensions with this checkout's Python code.

Once the dependencies and compilers are ready, the following is the core build
command used by the integration, adapted to a generic checkout. It is not a
complete fresh-machine provisioning script:

```
unset VLLM_PRECOMPILED_WHEEL_LOCATION
export VLLM_USE_PRECOMPILED=0
export VLLM_USE_PRECOMPILED_RUST=0
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=4
export NVCC_THREADS=1
export CARGO_BUILD_JOBS=2
export CMAKE_BUILD_TYPE=Release
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
.venv/bin/python setup.py bdist_wheel --dist-dir dist --py-limited-api=cp38
```

MAX_JOBS and CARGO_BUILD_JOBS limit build parallelism; reduce them for constrained
memory. The explicit compiler paths must exist in the chosen builder. Unsetting
the precompiled wheel location prevents accidentally packaging upstream binary
extensions instead of compiling the integration. Install the result into a
separate runtime environment before verification.

## Serving example

After building/installing the integration and independently obtaining a
compatible GLM-5.3-Flash AWQ W4A16 checkpoint, a core command is:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/GLM-5.3-Flash-AWQ-W4A16 \
  --host 127.0.0.1 --port 8000 \
  --served-model-name glm-5.3-flash-awq-rebased \
  --tensor-parallel-size 4 \
  --kv-cache-dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45
```

This example uses four selected visible GPUs, BF16 attention KV, the checkpoint's
W4A16 weights, and no MTP configuration. It is a core invocation, not the complete
private deployment or a promise of performance on arbitrary machines. Adapt
device selection, resource limits, cache directories, authentication, and graph
settings to the target environment. A noexec temporary directory can prevent
TileLang-generated shared libraries from loading; provide an appropriate
executable temporary directory inside the isolated runtime.

For optional deterministic Top-K and legacy W4A8 behavior, read
[the compatibility guide](../features/quantization/backport_sm80.md) before
changing defaults. Do not reuse packed weights across incompatible activation
layouts.
