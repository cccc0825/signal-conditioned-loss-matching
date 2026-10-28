"""Synthetic unit checks only; these are not robot results."""
import json
from pathlib import Path
import numpy as np
from goodpoints import compressc, kt
from goodpoints.compress import compresspp_kt
from skt_wrapper import select
root = Path(__file__).resolve().parent
rng = np.random.default_rng(72031)
checks=[]
for q in (1, 7):
    X=rng.normal(size=(1024, 21)); Y=rng.normal(size=(1024, q)); scale=4.2
    Z=np.ascontiguousarray(np.c_[X,Y]); params=np.array([scale**2,float(q)])
    K=np.empty((1024,1024))
    compressc.compute_K(Z,np.arange(1024,dtype=np.int64),b'loss_gaussian',params,K)
    sq=((X[:,None,:]-X[None,:,:])**2).sum(axis=2)
    base=np.exp(-sq/(2*scale**2)); expected=base**2+base*(Y@Y.T)
    err=float(np.max(np.abs(K-expected)))
    assert err<1e-12
    for m in (64,128,256):
        got, meta=select(X,Y,scale,m,17)
        direct=kt.thin_K(K,K,int(np.log2(1024//m)),delta=.5,seed=17,unique=True,mean0=False)
        assert np.array_equal(got,direct) and len(np.unique(got))==m
        checks.append(dict(kind='synthetic_unit_only',kernel_max_error=err,exact_index_parity=True,**meta))
    got,meta=select(X,Y,scale,32,17,mode='compresspp',unique=False)
    direct=compresspp_kt(Z,b'loss_gaussian',params,g=0,num_bins=4,delta=.5,seed=17,mean0=False)
    assert np.array_equal(got,direct)
    checks.append(dict(kind='synthetic_unit_only',kernel_max_error=err,exact_index_parity=True,**meta))
(root/'parity.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
