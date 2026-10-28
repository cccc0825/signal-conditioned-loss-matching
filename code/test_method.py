"""Synthetic algebra/unit tests only; these data never enter paper experiments."""
import numpy as np,json
from pathlib import Path
from method import weighted_matching,allocation,stratified,loss_kernel
from legacy_core import matching,kernel
from prepare_development import features
rng=np.random.default_rng(101);X=rng.normal(size=(32,6));Y=rng.normal(size=(32,2));K=kernel(X,X,1);H=loss_kernel(K,Y)
w=rng.random(32);w/=w.sum();idx=weighted_matching(H,w,8)
for t in range(8):
 chosen=idx[:t];n=t+1;vals=[]
 for j in range(32):
  if j in chosen: vals.append(np.inf);continue
  s=np.r_[chosen,j];vals.append(float(w@H@w+H[np.ix_(s,s)].mean()-2*np.mean((H@w)[s])))
 assert idx[t]==np.argmin(vals)
assert np.array_equal(weighted_matching(H,np.ones(32),8),matching(H,8))
for m in [1,7,31,32]:
 q,_=allocation(np.arange(32)%4,np.ones(32),m,1);assert q.sum()==m and np.all(q<=8)
a=np.zeros((40,6));a[20:,4:]=10;pos=np.r_[np.arange(20),np.arange(100,120)];f=features(a,pos,np.ones(2),np.ones(2));assert np.all(f[:20,0]==0)
print(json.dumps({'status':'pass','tests':['exact weighted MMD greedy objective','uniform target reduction','quota conservation/capacity','no smoothing across source-index gap'],'synthetic_only':True}))
