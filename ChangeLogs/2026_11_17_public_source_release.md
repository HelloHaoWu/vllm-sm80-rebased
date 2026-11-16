# Public source publication preparation

Date: 2026-11-17, Asia/Shanghai.

## Source baseline

Prepared from the validated integration source
88a7d60756d5d922f8771f6df32032210b877301, based on vLLM main
cfd86e43206515956029d5e5cd25ac7bb212da2c and vLLM Backport
29e66dad461b11caae6039f510eaa7913f9dfb19.

## Publication changes

- Remove private experimental ChangeLogs and the local trial README from the
  publication branch's migration history. Keep the original repository intact.
- Preserve upstream/fork authorship, Apache-2.0 licensing, inherited headers,
  and the original project READMEs as clearly labelled reference material.
- Add English and Chinese entry points, architecture/source ownership,
  validation limits, build guidance, explicit acknowledgements, and a
  source-tree provenance manifest.
- Make no runtime, kernel, test, build-recipe, or dependency changes.
- Distribute source only. Private model files, credentials, internal deployment
  scripts/logs, wheel files, and container images are not part of this publication.

## Verification scope

Publication verification compares runtime/build/test tree identities with the
validated source and checks the publication delta, reachable migration history,
documentation links, and license identity. The historical GPU/model results
are summarized in the validation document; preparing this source branch does
not constitute a fresh GPU run, wheel build, or formal model-quality evaluation.

See [acknowledgements](../ACKNOWLEDGEMENTS.md),
[architecture](../docs/backport_rebase/architecture.md),
[validation](../docs/backport_rebase/validation.md), and
[provenance](../docs/backport_rebase/provenance.json).
