# Author SKT comparison

The frozen aligned comparison uses the unmodified author `goodpoints` Cython KT-SPLIT + KT-SWAP algorithm on shared 1,024-candidate pools. Revision: `c2450234fb9eb86ef90dca037667a953bd3830c9`, source [ag2435/goodpoints](https://github.com/ag2435/goodpoints), MIT license. The exact archive is included; SHA-256 `68997a0e240d9cf873737bbff7d8c8cec601c5d5589f21c6cde9421e34205ca7`.

The adapter uses `delta=0.5`, `unique=True`, and 64/128/256 selected points. It explicitly configures the author loss-kernel product dimension for joint outputs and rescales standardized Y by sqrt(2/J). The Gaussian scale is the paper's scale times sqrt(d). `test_benchmark_mapping.py` checks this mapping and `test_parity.py` checks direct calls against the wrapper.

This is an explicitly configured multi-output direct-KT comparison. It is not the default scalar-output NPR estimator or the complete NPR Compress++ pipeline. In particular, the unmodified Compress++ path on 1,024 candidates outputs 32 points; it is not silently called a 64/128/256-point comparison. The wrapper rejects incompatible pool/budget combinations.

Optional rebuild (requires a C/C++ toolchain):

```bash
python3.12 -m venv skt-venv
skt-venv/bin/python -m pip install numpy==2.3.5 Cython==3.3.0 setuptools==84.0.0 wheel==0.48.0
skt-venv/bin/python -m zipfile -e skt_baseline/author_goodpoints.zip skt-source
skt-venv/bin/python -m pip install --no-build-isolation ./skt-source/goodpoints-c2450234fb9eb86ef90dca037667a953bd3830c9
python run.py extract --output runs/skt-unit-check
skt-venv/bin/python runs/skt-unit-check/skt_baseline/test_parity.py
skt-venv/bin/python runs/skt-unit-check/skt_baseline/test_benchmark_mapping.py
python run.py selectors --skt-python "$(pwd)/skt-venv/bin/python"
```

Run the two tests in the extracted copy, as shown: they write result JSON files, so running them directly in the release tree would change manifest-protected reference files. Use a new extraction path if `runs/skt-unit-check` already exists.

The original recorded author environment was Python 3.12.14 with the above package versions and goodpoints 0.2.5. A different compiler/platform can produce numeric differences; the frozen dictionaries remain available for prediction-level reproduction without compiling this dependency.

Related upstream NPR repository: [albertgong1/npr](https://github.com/albertgong1/npr), revision `dd5347b2568c006db2f92249c83ccf1ded15d015`, BSD-3-Clause. NPR source is not vendored here; its identity is retained to explain the original integration and the exact comparison boundary. GEMSS-MO in `legacy_core.py` is a local multi-output adaptation, not a claim to have reproduced the complete original GEMSS method.
