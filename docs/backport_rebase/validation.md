# Validation and performance scope

Evidence summarized here was collected by the maintainer through 2026-11-16,
before preparing this public source branch on 2026-11-17. Private hosts,
credentials, infrastructure paths, and raw deployment logs are not distributed.
This is a reported validation summary, not a publicly rerun benchmark artifact.

## Artifact and correctness validation

- Complete SM80 wheel and runtime image built from the integration source,
  including native CUDA/C++ and Rust components. Installed-file hashes and
  native-library identities were audited.
- The base complete build had 375 passing checks and one ROCm-conditional skip.
  The final Python index-width repair had four real sparse-attention and CUDA
  graph regressions pass. Native libraries were hash-identical between those
  two stages; this does not mean all 375 checks were rerun on the final package.
- Real four-A800 GLM startup, ordinary chat, automatic tool calls, tool-result
  round trips, structured JSON, and a separate single-token MTP experiment passed.
- The public branch has identical runtime/build-source blobs to the validated
  integration. Publication removes private notes and adds documentation; rewritten
  history changes some commit IDs and Git-derived version labels. The public
  commit has not yet been rebuilt into a new distributable wheel or image.

The tested build environment was Linux x86_64, Ubuntu 24.04, glibc 2.39,
GCC 13.3, CUDA 13.0, Python 3.12, Torch 2.14.0+cu130, Triton 3.8.0, and
Transformers 5.22.2. The old deployment used Torch 2.13.0+cu130, Triton 3.7.1,
and Transformers 5.17.0.

## Matched old/new deployment comparison

Both stacks used the same checkpoint and four A800 GPUs, TP4, BF16 attention KV,
0.85 GPU-memory utilization, batch-token budget 8192, maximum scheduled sequences
128, prefix caching, chunked prefill, asynchronous scheduling, and matched CUDA
graph settings. The configured context limit was 1,048,576; actual tested inputs
were 1,036, 8,204, and 32,780 tokens. Each request generated 512 tokens.

Four combinations, old/new with MTP off/on, each contributed two rounds:
72 timed and 12 warmup requests per combination, 336 requests overall, with zero
failed requests. The table averages per-round aggregate output throughput from
timed requests. Extra unmatched rounds are excluded. MTP used one draft token
per step on both stacks.

| Input tier | Concurrency | MTP off: old TPS | MTP off: new TPS | Change | MTP on: old TPS | MTP on: new TPS | Change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1K | 1 | 101.08 | 103.35 | +2.24% | 157.80 | 161.76 | +2.51% |
| 1K | 4 | 302.12 | 315.97 | +4.58% | 413.11 | 435.77 | +5.49% |
| 8K | 1 | 85.15 | 86.92 | +2.08% | 125.26 | 127.34 | +1.66% |
| 8K | 4 | 196.59 | 202.15 | +2.83% | 244.10 | 248.17 | +1.67% |
| 32K | 1 | 58.76 | 59.60 | +1.43% | 73.27 | 75.54 | +3.10% |
| 32K | 4 | 96.71 | 98.44 | +1.79% | 102.56 | 106.26 | +3.61% |

TPS denotes aggregate output tokens per second, including reasoning tokens and
end-to-end input/queue overhead, not per-GPU speed or sustained capacity. The
percentages use underlying unrounded measurements. Dependency versions differ,
so these are net deployment-stack changes, not a causal attribution to Git
rebase alone or to an individual kernel. Runs were grouped by configuration,
with only two matched rounds rather than a long randomized experiment.

MTP already accelerated the old stack. A roughly 60% gain obtained by toggling
MTP must not be presented as the rebase's unique benefit. Some latency measures
also regressed: with MTP on at 8K/concurrency 1, TTFT rose from about 0.963 to
1.044 seconds despite improved output throughput.

## Current-version concurrency sample

A separate MTP-off campaign used 2K input and 512 output tokens, with one timed
wave each at concurrency 8, 16, and 32. All 56 timed and 56 warmup requests
succeeded. Aggregate throughput was 423.79, 601.36, and 747.22 tokens/s;
per-request median decode speed was 64.69, 45.20, and 28.95 tokens/s.

This was not a matched old/new high-concurrency comparison. Thinking/sampling
settings differed from the matched campaign. All timed outputs exhausted the
512-token budget without final-answer content, so HTTP success and output TPS
are not evidence of completed Agent tasks. This sample does not establish
long-context concurrency, long-running stability, or maximum service capacity.

## Not established

- No formal UnityBench/ABV quality rerun for the new integration.
- No proof of score parity with the old stack or the official FP8 checkpoint.
- No actual one-million-token input validation.
- No complete service validation of every migrated model, PP/EP combination,
  LMCache deployment, legacy W4A8 model, or newer GPU architecture.
- No public downloadable wheel/container release yet.
