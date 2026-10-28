"""Finite development search and all out-of-record pilot predictions are archived."""
from pathlib import Path
import json,time,sys,hashlib,resource
import numpy as np
import legacy_core as core
from method import assign_bins,stratified,loss_kernel
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reference'
def run():
 started=time.time();rows=[];choices={}
 for robot in ['PANDA','MELFA']:
  tr=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));v=dict(np.load(ROOT/'data'/f'{robot}_validation.npz'));hp=json.loads((BASE/'results'/f'choices_{robot}.json').read_text())[robot+'_se_common'];pred=np.zeros_like(tr['Y']);folds=[]
  for g,name in enumerate(tr['names']):
   pool=np.flatnonzero(tr['group']!=g);ix=np.random.default_rng(7300+g).permutation(pool)[:256];held=np.flatnonzero(tr['group']==g)
   # Training-only scaling per pilot fold; no held-record targets in pilot fit.
   mu=tr['X'][pool].mean(0);sd=tr['X'][pool].std(0);ym=tr['Y'][pool].mean(0);ys=tr['Y'][pool].std(0)
   X=(tr['X'][ix]-mu)/sd;V=(tr['X'][held]-mu)/sd;Y=(tr['Y'][ix]-ym)/ys
   f,_=core.fit_predict(core.kernel(X,X,hp['scale']),core.kernel(V,X,hp['scale']),Y,hp['lambda_mean']);pred[held]=f*ys+ym
   folds.append(dict(record=str(name),fit_indices=ix.tolist(),held_indices=held.tolist(),seed=7300+g));assert not np.intersect1d(ix,held).size
  error=np.mean(((pred-tr['Y'])/tr['ys'])**2,axis=1);np.savez_compressed(ROOT/'results'/f'{robot}_pilot.npz',pred_physical=pred,error=error);(ROOT/'results'/f'{robot}_pilot_folds.json').write_text(json.dumps(folds,indent=2))
  X=(tr['X']-tr['mu'])/tr['sd'];Y=(tr['Y']-tr['ym'])/tr['ys'];V=(v['X']-tr['mu'])/tr['sd'];VY=(v['Y']-tr['ym'])/tr['ys'];K=core.kernel(X,X,hp['scale']);H=loss_kernel(K,Y);VK=core.kernel(V,X,hp['scale']);out=ROOT/'results/development_selections';out.mkdir(exist_ok=True)
  for kind in ['energy','joint']:
   bins,cut=assign_bins(tr['Z'],kind)
   for power in [.5,1.,2.]:
    t=time.perf_counter();ix,meta=stratified(H,bins,error,256,power);f,_=core.fit_predict(K[np.ix_(ix,ix)],VK[:,ix],Y[ix],hp['lambda_mean']);score=core.grouped_error(f,VY,v['group']);tag=f'{robot}_{kind}_{power}';np.save(out/(tag+'.npy'),ix)
    row=dict(robot=robot,kind=kind,power=power,cut=cut.tolist(),validation_nmse=score,selection_seconds=time.perf_counter()-t,allocation=meta,tag=tag);rows.append(row);print(json.dumps(row),flush=True)
  best=min([a for a in rows if a['robot']==robot],key=lambda a:a['validation_nmse']);choices[robot]={**best,'predictor':hp}
 (ROOT/'results/development_search.json').write_text(json.dumps(rows,indent=2));(ROOT/'config/method_choices.json').write_text(json.dumps(choices,indent=2));(ROOT/'results/development_run.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-started,peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,python=sys.version,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':run()
