"""Second pre-budgeted family: learned signal target with global centered-loss matching."""
from pathlib import Path
import json,time
import numpy as np
import legacy_core as core
from method import assign_bins,allocation,weighted_matching,loss_kernel
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reference'
rows=[];choices={}
for robot in ['PANDA','MELFA']:
 tr=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));v=dict(np.load(ROOT/'data'/f'{robot}_validation.npz'));hp=json.loads((BASE/'results'/f'choices_{robot}.json').read_text())[robot+'_se_common'];error=np.load(ROOT/'results'/f'{robot}_pilot.npz')['error']
 X=(tr['X']-tr['mu'])/tr['sd'];Y=(tr['Y']-tr['ym'])/tr['ys'];V=(v['X']-tr['mu'])/tr['sd'];VY=(v['Y']-tr['ym'])/tr['ys'];K=core.kernel(X,X,hp['scale']);H=loss_kernel(K,Y);VK=core.kernel(V,X,hp['scale'])
 for kind in ['energy','joint']:
  bins,cut=assign_bins(tr['Z'],kind)
  for power in [.5,1.,2.]:
   _,meta=allocation(bins,error,256,power);w=np.asarray(meta['target'])[bins]/np.asarray(meta['counts'])[bins];t=time.perf_counter();ix=weighted_matching(H,w,256);f,_=core.fit_predict(K[np.ix_(ix,ix)],VK[:,ix],Y[ix],hp['lambda_mean']);score=core.grouped_error(f,VY,v['group']);tag=f'{robot}_weighted_{kind}_{power}';np.save(ROOT/'results/development_selections'/(tag+'.npy'),ix)
   row=dict(robot=robot,family='weighted',kind=kind,power=power,cut=cut.tolist(),validation_nmse=score,selection_seconds=time.perf_counter()-t,allocation=meta,tag=tag);rows.append(row);print(json.dumps(row),flush=True)
 old=[dict(a,family='quota') for a in json.loads((ROOT/'results/development_search.json').read_text()) if a['robot']==robot]
 best=min(old+[a for a in rows if a['robot']==robot],key=lambda a:a['validation_nmse']);choices[robot]={**best,'predictor':hp}
(ROOT/'results/development_weighted_search.json').write_text(json.dumps(rows,indent=2));(ROOT/'config/final_method_choices.json').write_text(json.dumps(choices,indent=2))
