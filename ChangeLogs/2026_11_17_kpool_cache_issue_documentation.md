# Document the historical KPool indexer-cache corruption

Date: 2026-11-17, Asia/Shanghai.

Baseline: public branch main at 6fbdf41c507b776f5fd8c6d62aa8aaa53bd61824.

## Changes

- Add a focused KPool issue note with an English summary and Chinese explanation.
- Explain padded-stride addressing, the affected indexer keys/scales, possible
  prefix-cache symptoms, and the distinction from main MLA KV storage.
- Attribute the addressing repair to vLLM PR #57477 and its upstream contributors.
- Separate that historical defect from the later 2048/2176 remapping-width
  integration repair. Do not claim complete model-score recovery.
- Link the note from the English and Chinese READMEs and the architecture guide.

## Verification

Check documentation links, diff whitespace, upstream provenance, and unchanged
runtime/build/test source identities. This documentation update does not run GPU
tests, rebuild artifacts, or change any running model service. Historical test
results remain labelled as historical in the issue note.

[Issue explanation](../docs/backport_rebase/kpool_cache_corruption.md)
