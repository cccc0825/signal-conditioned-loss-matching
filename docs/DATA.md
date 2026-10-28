# Data source and processing

Source: **Robot Inverse Dynamics Dataset**, DOI [10.5281/zenodo.12516500](https://doi.org/10.5281/zenodo.12516500). Authors: Giulio Giacomuzzo, Ruggero Carli, Diego Romeres and Alberto Dalla Libera; Mitsubishi Electric Research Laboratories (MERL). License: [Creative Commons Attribution–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

This release uses the measured `Robots/PANDA` and `Robots/MELFA_RV4FRL` trajectories from LIP4RID. It does not use the simulated-Panda portion as real observations. The distributed arrays and frozen selections are adaptations of the source data and are distributed under the same CC-BY-SA 4.0 license. Changes comprise filtered state/torque column extraction, every-tenth-row sampling, historical record splits, training-only normalization, gap-aware derivative-energy descriptors, pilot predictions and selected indices.

The raw `LIP4RID.zip` is 541,538,288 bytes. Its SHA-256 is `527ec997b8f957f102e07f132f2663156ffcfe9366ed1c9f1d6e14ad21409288`. The raw archive is not required for the normal reproduction command. Obtain it from the linked Zenodo record for source-level inspection; do not substitute a different version silently.

After `python run.py extract`, inspect:

- `reference/research/dataset_README.md`: upstream README.
- `reference/research/lip_data_manifest.json`: source member, SHA-256, filtered columns and historical split for each record.
- `data/*_train.npz`, `data/*_validation.npz`, `reference/data/*/test.npz`: historical development/evaluation arrays.
- `data/MELFA_holdout.npz`: four separately frozen additional source records.
- `research/holdout_membership.json`, `research/holdout_quality_ledger.json`: exact additional membership and quality outcomes.
- `research/protocol.md`, `research/development_scope.md`: historical protocol and known exposure.

`research/holdout_membership.json` is a historical snapshot: `payload_unopened=true` describes the membership-freeze time on 2026-09-15. The data have since been opened for evaluation and are now included in this public reproduction package.

The additional records are `MELFA_range0.5_seed6`, `MELFA_range1.4_seed0`, `MELFA_range1.4_seed10`, and `MELFA_range1.4_seed5` from the same source archive. Other-amplitude development records overlap seed labels 6, 0 and 10; seed 5 has no matching historical seed label. These are additional records, not evidence of four independent motion-generating populations. The evaluation aggregates seeds within records and then averages records, rather than treating all sampled rows as independent experiments.

The offline signal descriptor uses a centered 31-original-row moving average of normalized squared joint velocity and acceleration. Windows stop at nonconsecutive source dataframe indices and use nearest-endpoint extension inside each segment. No temporal sampling rate or physical Hz is inferred for recordings without timestamps. The predictor only uses q, dq and ddq.

The supplied processed arrays are the input boundary of the tested one-command replay. `code/source_reader.py` supplies the raw archive reader and historical array conversion; `code/prepare_development.py` reconstructs the training/validation descriptors if `raw/LIP4RID.zip` is supplied in an extracted working directory. Exact raw-to-holdout reconstruction is documented by the membership and quality ledger; the public one-command replay does not claim to perform a new raw-to-results experiment.
