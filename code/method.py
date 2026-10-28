"""Cross-record error allocation with within-condition centered-loss matching."""
import numpy as np
from legacy_core import matching

def assign_bins(Z,kind,cut=None):
 if kind=='energy':
  cut=np.quantile(Z[:,0],[.25,.5,.75]) if cut is None else np.asarray(cut)
  b=np.digitize(Z[:,0],cut)
 elif kind=='joint':
  cut=np.median(Z,axis=0) if cut is None else np.asarray(cut)
  b=(Z[:,0]>cut[0]).astype(int)+2*(Z[:,1]>cut[1]).astype(int)
 else:raise ValueError(kind)
 return b,cut

def allocation(bins,error,m,power=.5):
 n=len(bins);counts=np.bincount(bins,minlength=4);global_error=float(np.mean(error))
 # 32 pseudo-observations of the pooled error stabilise small regimes.
 errors=(np.bincount(bins,weights=error,minlength=4)+32*global_error)/(counts+32)
 relative=np.clip(errors/max(global_error,1e-12),.25,4.)
 target=counts/n*relative**power;target/=target.sum()
 desired=m*target;quota=np.minimum(np.floor(desired).astype(int),counts)
 while quota.sum()<m:
  eligible=quota<counts;v=int(np.argmax(np.where(eligible,desired-quota,-np.inf)));quota[v]+=1
 return quota,dict(counts=counts.tolist(),errors=errors.tolist(),target=target.tolist(),quota=quota.tolist(),power=power)

def stratified(H,bins,error,m,power,random_within=False,seed=0):
 quota,meta=allocation(bins,error,m,power);out=[];rng=np.random.default_rng(seed)
 for b in range(4):
  ids=np.flatnonzero(bins==b);q=int(quota[b])
  if q:out.extend(ids[rng.permutation(len(ids))[:q] if random_within else matching(H[np.ix_(ids,ids)],q)])
 ids=np.asarray(out,int);assert len(ids)==m and len(np.unique(ids))==m
 return ids,meta

def loss_kernel(K,Y,full=False):
 H=K*K+2/Y.shape[1]*K*(Y@Y.T)
 if full:H+=(Y*Y)@(Y*Y).T/Y.shape[1]
 return H

def weighted_matching(H,w,m):
 w=np.maximum(w,0);w=w/w.sum();mu=H@w;diag=H.diagonal();summed=np.zeros(len(H));used=np.zeros(len(H),bool);out=[]
 for t in range(m):
  a=diag-2*(t+1)*mu+2*summed;a[used]=np.inf;i=int(a.argmin());out.append(i);used[i]=True;summed+=H[:,i]
 return np.asarray(out,int)
