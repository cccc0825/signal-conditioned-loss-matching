"""Synthetic mapping test; not a robot experiment."""
import json
from pathlib import Path
import numpy as np
from goodpoints import compressc
rng=np.random.default_rng(240915)
X=rng.normal(size=(64,21));Y=rng.normal(size=(64,7));s=0.8
d=X.shape[1];q=Y.shape[1]
Z=np.ascontiguousarray(np.c_[X,Y*np.sqrt(2/q)])
K=np.empty((64,64))
compressc.compute_K(Z,np.arange(64,dtype=np.int64),b'loss_gaussian',np.array([d*s*s,float(q)]),K)
base=np.exp(-((X[:,None]-X[None,:])**2).sum(axis=2)/(2*d*s*s))
H=base**2+(2/q)*base*(Y@Y.T)
err=float(np.max(np.abs(K-H)))
assert err<1e-12
Path(__file__).with_name('benchmark_mapping.json').write_text(json.dumps({'kind':'synthetic_unit_only','d':d,'q':q,'s':s,'wrapper_scale':'s*sqrt(d)','wrapper_labels':'Y*sqrt(2/q)','max_absolute_gram_error':err,'pass':True},indent=2))
print(err)
