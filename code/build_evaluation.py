"""Freeze comparison selections and equally sized secondary predictor grids before holdout."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import legacy_core as core
from method import assign_bins,allocation,weighted_matching,loss_kernel
ROOT=Path(__file__).resolve().parents[1]
EVAL=list(range(7310,7320));DEV=[7320,7321,7322]

def main():
 choices=json.loads((ROOT/'config/final_method_choices.json').read_text());allmeta=[];retuned={};tuning=[]
 for robot,ch in choices.items():
  tr=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));v=dict(np.load(ROOT/'data'/f'{robot}_validation.npz'));e=np.load(ROOT/'results'/f'{robot}_pilot.npz')['error'];hp=ch['predictor'];n=len(e);out=ROOT/'results/selections'/robot;out.mkdir(parents=True,exist_ok=True)
  X=(tr['X']-tr['mu'])/tr['sd'];Y=(tr['Y']-tr['ym'])/tr['ys'];V=(v['X']-tr['mu'])/tr['sd'];VY=(v['Y']-tr['ym'])/tr['ys'];t=time.perf_counter();K=core.kernel(X,X,hp['scale']);H=loss_kernel(K,Y);HF=loss_kernel(K,Y,True);build=time.perf_counter()-t
  bins,cut=assign_bins(tr['Z'],ch['kind'],ch['cut']);_,am=allocation(bins,e,256,ch['power']);w=np.asarray(am['target'])[bins]/np.asarray(am['counts'])[bins]
  np.savez_compressed(ROOT/'results'/f'{robot}_target.npz',bins=bins,weights=w,error=e)
  individual=np.clip(e/max(e.mean(),1e-12),.25,4.)**ch['power'];static=(1+bins).astype(float)
  # Negative control preserves exact bin counts but destroys alignment with signal conditions.
  rbin=bins[np.random.default_rng(7341).permutation(n)];_,rm=allocation(rbin,e,256,ch['power']);rw=np.asarray(rm['target'])[rbin]/np.asarray(rm['counts'])[rbin]
  methods=['random','kcenter','variance','gemss_mo','centered','full','signal_task','residual_only','random_bins','signal_static','weighted_random']
  for method in methods:
   stochastic=method in ['random','kcenter','gemss_mo','weighted_random'];seeds=EVAL+DEV if stochastic else [0]
   for seed in seeds:
    t=time.perf_counter()
    if method=='random':idx=np.random.default_rng(seed).permutation(n)[:256]
    elif method=='kcenter':idx=core.kcenter(X,256,seed)
    elif method in ['variance','gemss_mo']:idx=core.adaptive(K,Y,256,256*hp['lambda_mean'],method,seed)
    elif method=='centered':idx=core.matching(H,256)
    elif method=='full':idx=core.matching(HF,256)
    elif method=='signal_task':idx=weighted_matching(H,w,256)
    elif method=='residual_only':idx=weighted_matching(H,individual,256)
    elif method=='random_bins':idx=weighted_matching(H,rw,256)
    elif method=='signal_static':idx=weighted_matching(H,static,256)
    elif method=='weighted_random':idx=np.random.default_rng(seed).choice(n,256,replace=False,p=w/w.sum())
    cost=time.perf_counter()-t;np.save(out/f'{method}_{seed}.npy',idx);allmeta.append(dict(robot=robot,method=method,seed=seed,selection_seconds=cost,kernel_build_seconds=build,indices_sha256=hashlib.sha256(idx.tobytes()).hexdigest()))
   # Equal 3x3 predictor grid, selection kernel and dictionaries remain fixed.
   ids=[np.load(out/f'{method}_{s}.npy') for s in (DEV if stochastic else [0])]
   grid=[]
   for sf in [.5,1.,2.]:
    for lf in [.1,1.,10.]:
     errs=[]
     for ix in ids:
      f,_=core.fit_predict(core.kernel(X[ix],X[ix],hp['scale']*sf),core.kernel(V,X[ix],hp['scale']*sf),Y[ix],hp['lambda_mean']*lf);errs.append(core.grouped_error(f,VY,v['group']))
     z=dict(robot=robot,method=method,scale=hp['scale']*sf,lambda_mean=hp['lambda_mean']*lf,val_nmse=float(np.mean(errs)));grid.append(z);tuning.append(z)
   retuned[f'{robot}_{method}']=min(grid,key=lambda a:a['val_nmse']);print(robot,method,retuned[f'{robot}_{method}']['val_nmse'],flush=True)
  # Exact algebraic reduction is tested on the actual development kernel.
  assert np.array_equal(weighted_matching(H,np.ones(n),64),core.matching(H,64))
 (ROOT/'results/selection_times.json').write_text(json.dumps(allmeta,indent=2));(ROOT/'results/secondary_tuning.json').write_text(json.dumps(tuning,indent=2));(ROOT/'config/secondary_predictor_choices.json').write_text(json.dumps(retuned,indent=2))
if __name__=='__main__':main()
