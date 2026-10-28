# Signal-Conditioned Loss Matching for Robot Inverse Dynamics Compression

Research code and frozen evaluation inputs for the paper **Signal-Conditioned Loss Matching for Robot Inverse Dynamics Compression**.

The study asks when matching a loss-related kernel better translates into better robot inverse-dynamics prediction. It retains the loss-matching analysis and uses training-record cross-fitted residuals together with local velocity/acceleration energy to condition dictionary selection. The prediction model remains state-only kernel ridge regression (KRR).

## Reproduce the reported evaluation

Use Python 3.12. The main replay runs on CPU and needs no raw dataset download or optional Cython baseline installation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py verify
python run.py reproduce
```

Each run creates a fresh `runs/<UTC timestamp>/` directory. The program verifies the archived inputs, fits every frozen dictionary, writes **771 prediction files**, regenerates record-level metrics, tables and figures, and compares all **7,453 metric rows** to the archived reference values. A mismatch makes the command fail. See `validation.json`, `logs/`, `analysis/`, `figures/`, and `paper/` in the output directory. The `paper/` files are automatically generated numerical table fragments, not a manuscript template.

The replay reconstructs predictions from frozen selected indices; it does not merely copy existing predictions. It includes all reported budgets (64, 128, 256), shared/retuned predictors, baselines, ablations, retrospective tests and four additional MELFA records. Runtime and numeric differences depend on hardware and BLAS; the verification tolerance is `atol=rtol=1e-10`.

The public package was checked on 2026-10-28: all 771 predictions and 7,453 metric rows matched the archived values exactly (maximum absolute difference 0.0); 196 rebuilt non-SKT dictionary files matched exactly. See [docs/reproduction_check.json](docs/reproduction_check.json) for the tested scope.

## Rebuild the selections

```bash
python run.py selectors
python run.py selectors --development
```

The first command rebuilds and checks **118 primary dictionaries and 78 aligned non-SKT dictionaries**. The second also repeats the two families of validation-based rule selection (24 configurations across two robots). These steps use training and validation data only. They are replays of historical development, not a newly registered study. Allocation weights and choices are available in the archived JSON and NPZ files.

To rebuild the additional **60 author-SKT dictionaries**, install the archived author implementation in a separate Python 3.12 environment, as described in [docs/SKT.md](docs/SKT.md), and run:

```bash
python run.py selectors --skt-python /absolute/path/to/skt-venv/bin/python
```

Without `--skt-python`, the author-SKT selections remain frozen inputs. Their KRR predictions and evaluation are still rebuilt by `reproduce`.

## What the results support

On the four additional MELFA records, with a 256-point dictionary and the shared predictor settings, signal-conditioned selection gives record-averaged NMSE **0.2289**, versus **0.2531** for random selection: a **9.6% reduction**. Physical torque RMSE is **5.064 Nm** versus **5.212 Nm**.

This is a conditional result. At the same primary setting, k-center and GP variance have lower NMSE than the proposed rule. The advantage over random reverses at budget 64, and retuning changes the ordering. The aligned 1,024-candidate author-SKT comparison and the ablations are included in the replay. Small differences and leave-one-record-out sensitivity are retained; no records are removed to improve the result.

The four additional records come from the **same source archive**, not an independently collected robot experiment. Three excitation seed labels appear at other amplitudes in previously exposed development records. Historical retrospective tests retain their original exposure history. See [docs/DATA.md](docs/DATA.md) and the archived protocol for the exact membership, quality checks and limits.

## Repository contents

- `code/`: signal descriptors, rule development, selectors, KRR evaluation, aggregation and plotting.
- `artifacts/frozen_inputs.zip`: processed real-robot arrays, record memberships, frozen dictionaries, pilot predictions, historical development traces and reference metrics. Data contents follow CC-BY-SA 4.0.
- `config/`: selected settings and the public integrity manifest.
- `skt_baseline/`: explicit adapter, tests, provenance and the pinned MIT-licensed author archive.
- `docs/`: source/data provenance, rebuild instructions and public packaging changes.
- `licenses/`: third-party license notices.

`python run.py extract --output /path/to/new/directory` safely unpacks a working copy for inspection. The public integrity manifest describes this release; it does not replace the original study's chronology or claim a fresh preregistration. Large regenerated predictions, raw source archives, private working documents and machine environments are not included.

## Data and attribution

The real-robot data are derived from **Robot Inverse Dynamics Dataset (LIP4RID)** by **Giulio Giacomuzzo, Ruggero Carli, Diego Romeres and Alberto Dalla Libera**, released by Mitsubishi Electric Research Laboratories under **CC-BY-SA 4.0**: [Zenodo record and DOI](https://doi.org/10.5281/zenodo.12516500). The source dataset README is retained inside the input archive. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the data adaptation and baseline notices.

The repository is a research reproduction package. The experimental scope is offline dictionary compression and inverse-dynamics prediction; it does not include a hardware control experiment.
