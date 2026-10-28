# Quality-gate amendments

## v1 → v2, before candidate payload reveal

The original `holdout_quality_rules.md` and original `holdout_membership_freeze.json` remain unchanged. All four candidate members and their membership digest remain fixed.

Reason: the completed development method uses dimensionless velocity/acceleration energy descriptors over source-index windows, rather than physical time or spectral frequency. The development source code preserves a discontinuous but increasing DataFrame index, splits at gaps, and smooths each segment separately. Requiring a measured timestamp would be inconsistent with that method and with the exposed MELFA schema.

The v2 rules therefore make timestamp availability optional; require finite, strictly increasing source indices; split at every increment other than one; require at least one segment of 31 consecutive rows; and retain all segment edges through nearest-endpoint extension. Short segments also retain their rows under the frozen implementation when the record has at least one qualifying segment. No windows cross gaps. All four records, including the short seed-5 candidate, remain members.

The schema/finite checks, complete quality ledger, no target summaries, prohibition on performance-based exclusion, shared-seed limitations and requirement for a complete root method/endpoint freeze remain in force. No candidate measurement payload was opened and no candidate predictive outcome was observed to make this amendment.

Exact amendment time, source-code/configuration hashes, old/new rule hashes and the unchanged membership hash are in `quality_amendment_v2.json`. The root's final experimental freeze should bind `quality_v2_freeze.json` as the effective quality gate. Current status remains **quality sealed; waiting for final freeze notification**.
