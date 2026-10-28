"""CPU SKT baseline using unmodified author goodpoints Cython algorithms.

Inputs are candidate-only standardized X/Y. No fitting or holdout access occurs.
"""
from pathlib import Path
import hashlib
import time
import numpy as np
from goodpoints import compressc, kt
from goodpoints.compress import compresspp_kt

NPR_COMMIT = 'dd5347b2568c006db2f92249c83ccf1ded15d015'
GOODPOINTS_COMMIT = 'c2450234fb9eb86ef90dca037667a953bd3830c9'

def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def select(X, Y, scale, m, seed, *, mode='direct_kt', unique=True):
    """Return (indices, identity). m is output count, not halving rounds.

    direct_kt: exact author KT-SPLIT+KT-SWAP on its loss kernel. Requires
      N/m power of two. unique=True is an explicit no-repetition restriction.
    compresspp: unchanged author Compress++ path. Requires N power of four,
      m=sqrt(N), unique=False. This rejects upstream silent pool truncation.
    Multi-output uses author Cython product-dimension parameter q (NPR estimator
    hardcodes 1); this is explicit vector-output configuration, not a verified
    reproduction of paper multi-output experiments/theory.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y):
        raise ValueError('X/Y must be matching two-dimensional arrays')
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('finite inputs and positive scale required')
    n, q = len(X), Y.shape[1]
    if m < 1 or m > n or int(m) != m or int(seed) != seed or seed < 0:
        raise ValueError('invalid output count or seed')
    m, seed = int(m), int(seed)
    Z = np.ascontiguousarray(np.column_stack([X, Y]))
    params = np.array([float(scale)**2, float(q)])
    start = time.perf_counter()
    if mode == 'direct_kt':
        ratio = n // m
        if n % m or ratio & (ratio - 1):
            raise ValueError('direct KT requires N/m an exact power of two; provide a shared frozen pool')
        rounds = ratio.bit_length() - 1
        K = np.empty((n, n), dtype=np.float64)
        compressc.compute_K(Z, np.arange(n, dtype=np.int64), b'loss_gaussian', params, K)
        idx = kt.thin_K(K, K, rounds, delta=0.5, seed=seed, unique=unique, mean0=False)
        name = 'SKT loss kernel with author direct KT-SPLIT and KT-SWAP'
    elif mode == 'compresspp':
        if n & (n-1) or (n.bit_length()-1) % 2 or m*m != n:
            raise ValueError('author Compress++ requires power-of-four N and m=sqrt(N) in this wrapper')
        if unique:
            raise ValueError('author Compress++ does not expose unique; use unique=False and archive any repeats')
        idx = compresspp_kt(Z, b'loss_gaussian', params, g=0, num_bins=4, delta=0.5, seed=seed, mean0=False)
        rounds = None
        name = 'SKT author Compress++ selection path'
    else:
        raise ValueError('unknown mode')
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) != m or np.any(idx < 0) or np.any(idx >= n):
        raise RuntimeError('upstream output cardinality/range failure')
    identity = dict(method=name, mode=mode, n=n, m=m, q=q, seed=seed,
                    scale=float(scale), delta=0.5, unique_constraint=bool(unique),
                    distinct_indices=int(len(np.unique(idx))), halving_rounds=rounds,
                    kernel='exp(-||x-xprime||^2/(2*scale^2))^2 + dot(y,yprime)*exp(-||x-xprime||^2/(2*scale^2))',
                    npr_commit=NPR_COMMIT, goodpoints_commit=GOODPOINTS_COMMIT,
                    multioutput_configuration='author Cython product=q; NPR estimator hardcodes product=1',
                    input_sha256=_hash(Z), indices_sha256=_hash(idx),
                    selection_seconds=time.perf_counter()-start,
                    pool_reduction=False, gram_bytes=8*n*n if mode=='direct_kt' else None)
    return idx, identity
