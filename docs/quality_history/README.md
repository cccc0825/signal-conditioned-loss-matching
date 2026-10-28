# Historical holdout quality records

These are unchanged records from the September 2026 experiment, added to the public package on 2026-10-28. They preserve the original chronology and do not establish a new preregistration.

The v1 rules were superseded by the v2 source-row rules before the recorded candidate reveal. The four candidate members remain unchanged. Statements such as `payload_unopened`, `reveal_authorized: false` and “waiting for final freeze” describe the status at the historical recording time; the measurements have since been evaluated and are distributed in the frozen input archive.

## Paths in the historical records

- `holdout_membership_freeze.json` is included here unchanged. It has identical bytes to `research/holdout_membership.json` in `artifacts/frozen_inputs.zip`.
- The historical `work/signal_extension/data/MELFA_feature_config.json` is now `data/MELFA_feature_config.json` inside that archive, with unchanged bytes.
- The historical `work/signal_extension/code/prepare_development.py` corresponds to the public `code/prepare_development.py`. Public path adaptations changed its file hash; [source_changes.json](../source_changes.json) records the original and public hashes. Numerical processing is retained.
- The quality outcomes are `research/holdout_quality_ledger.json` inside the input archive. Use `python run.py extract` to inspect these archived paths.

The historical hash records below describe those historical files, not the current release launcher or the complete current repository.

| File | SHA-256 |
| --- | --- |
| `holdout_membership_freeze.json` | `9ce131de0030df9020934ad6263e61e985dbc3a10583bf19a20a104e1ce9d53a` |
| `holdout_quality_rules.md` | `9201e5919dad540556d6b363bcb70398f746978fc163a83cbaae5374726a9f41` |
| `holdout_quality_rules_v2.md` | `0dea0bc59063cda8ce8a83d10d6eb605a85638fdebf3fe961d3482bd59674882` |
| `quality_amendment_v2.json` | `b668480ce31544b4fa3a171c46518fbba5df3d7fb6c2c3f736f94ccda8c19097` |
| `quality_change_log.md` | `ff95309459488a0073eff88c62fdb36b9a46d1c4ae79c457ae126b2b5cb0c035` |
| `quality_v2_freeze.json` | `75fc2865d2359ff3367a3d3a6476a0561323ab3daac31a3a6a614e5388311c39` |
