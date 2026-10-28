"""Matched 1024-candidate direct-KT comparison, with pool-only 64-point pilots."""
from pathlib import Path
import json,time
import numpy as np
import legacy_core as core
from method import assign_bins,allocation,weighted_matching,loss_kernel
ROOT=Path(__file__).resolve().parents[1]
choices=json.loads((ROOT/'config/final_method_choices.json').read_text());meta=[]
for robot,ch in choices.items():
 t=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));ids=np.random.default_rng(7350).permutation(len(t['X']))[:1024];rawX=t['X'][ids];rawY=t['Y'][ids];groups=t['group'][ids];mu=rawX.mean(0);sd=rawX.std(0);ym=rawY.mean(0);ys=rawY.std(0);X=(rawX-mu)/sd;Y=(rawY-ym)/ys;hp=ch['predictor'];pred=np.zeros_like(rawY);folds=[]
 for g in np.unique(groups):
  pool=np.flatnonzero(groups!=g);held=np.flatnonzero(groups==g);ix=np.random.default_rng(7360+int(g)).permutation(pool)[:64];fm=rawX[pool].mean(0);fs=rawX[pool].std(0);fy=rawY[pool].mean(0);fys=rawY[pool].std(0)
  A=(rawX[ix]-fm)/fs;B=(rawX[held]-fm)/fs;P,_=core.fit_predict(core.kernel(A,A,hp['scale']),core.kernel(B,A,hp['scale']),(rawY[ix]-fy)/fys,hp['lambda_mean']);pred[held]=P*fys+fy;folds.append(dict(held_group=int(g),fit_indices=ix.tolist(),held_indices=held.tolist()))
 e=np.mean(((pred-rawY)/ys)**2,axis=1);bins,_=assign_bins(t['Z'][ids],ch['kind'],ch['cut']);_,a=allocation(bins,e,256,ch['power']);w=np.asarray(a['target'])[bins]/np.asarray(a['counts'])[bins]
 out=ROOT/'results/skt_aligned'/robot;out.mkdir(parents=True,exist_ok=True);K=core.kernel(X,X,hp['scale']);H=loss_kernel(K,Y);F=loss_kernel(K,Y,True)
 np.savez_compressed(out/'pool.npz',X=X,Y=Y,rawX=rawX,rawY=rawY,mu=mu,sd=sd,ym=ym,ys=ys,original_indices=ids,group=groups,Z=t['Z'][ids],pilot_pred=pred,error=e,target=w)
 np.savez_compressed(out/'skt_input.npz',X=X,Y=Y*np.sqrt(2/Y.shape[1]));(out/'folds.json').write_text(json.dumps(folds))
 for method in ['random','centered','full','signal_task']:
  for seed in (range(7310,7320) if method=='random' else [0]):
   st=time.perf_counter()
   if method=='random':idx=np.random.default_rng(seed).permutation(1024)[:256]
   elif method=='centered':idx=core.matching(H,256)
   elif method=='full':idx=core.matching(F,256)
   else:idx=weighted_matching(H,w,256)
   for m in [64,128,256]:np.save(out/f'{method}_{seed}_{m}.npy',idx[:m])
   meta.append(dict(robot=robot,method=method,seed=seed,selection_seconds=time.perf_counter()-st,computed_max_budget=256))
 (out/'skt_call.json').write_text(json.dumps(dict(scale=hp['scale']*np.sqrt(X.shape[1]),budgets=[64,128,256],seeds=list(range(7310,7320)),unique=True,delta=.5)))
(ROOT/'results/skt_aligned/core_times.json').write_text(json.dumps(meta,indent=2));print('Aligned candidate pools, pool-only pilots and baseline dictionaries saved.')
