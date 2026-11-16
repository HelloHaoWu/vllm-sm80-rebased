# Acknowledgements and provenance

This repository is an independent integration of two open-source projects:

- [vLLM](https://github.com/vllm-project/vllm), by the vLLM team and contributors,
  provides the serving engine, scheduler, cache management, model implementations,
  upstream operators, and build infrastructure.
- [vLLM Backport](https://github.com/wtdcode/vllm-backport), by
  [wtdcode / lazymio](https://github.com/wtdcode) and its contributors, provides
  the older-GPU adaptations, including SM80 software FP8 encoding/decoding,
  Triton sparse-indexer scoring, deterministic FP32 Top-K, legacy Marlin
  compatibility, and numerous model and performance extensions.

The work published here rebases and adapts those Backport changes to a pinned,
newer vLLM main snapshot, isolates compatibility operators, resolves integration
conflicts, and fixes build and sparse-index-buffer contracts. We do not claim
the original Backport kernels or upstream vLLM infrastructure as our inventions.

Other projects and kernels used transitively, including Triton, Marlin,
FlashAttention, FlashInfer, TileLang, and LMCache, retain their respective
licenses and attribution in the inherited sources and dependency definitions.
Model checkpoints are distributed separately under their own terms.

## Pinned inputs

| Input | Commit |
| --- | --- |
| vLLM main | cfd86e43206515956029d5e5cd25ac7bb212da2c |
| vLLM Backport migration input | 29e66dad461b11caae6039f510eaa7913f9dfb19 |
| Privately validated integration source | 88a7d60756d5d922f8771f6df32032210b877301 |

The integration-source hash identifies the pre-publication source; it is not a
commit that readers should expect to resolve in either upstream project.
The public branch removes private deployment notes from the migration history.
Consequently, some commit hashes change. Original upstream and Backport authors
and commit messages are retained. Runtime source-tree identities are recorded
in [provenance.json](docs/backport_rebase/provenance.json).

The repository retains the original [Apache-2.0 license](LICENSE), existing
copyright notices, third-party notices, and source headers. It is not an official
vLLM release or an endorsement by either upstream project. AI assistance was
used for the integration and documentation; the maintainer is responsible for
review and acceptance.
