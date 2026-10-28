"""Additional author-KT comparison on exactly matching candidate pools and dictionaries."""
from pathlib import Path
import json,time,hashlib
import numpy as np
import legacy_core as core
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reference'
fr=json.loads((ROOT/'config/public_manifest.json').read_text())
for n,h in fr['files'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
expected={n for n in fr['files'] if n.startswith('results/skt_aligned/') and n.endswith('.npy')};actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'results/skt_aligned').rglob('*.npy')};assert actual==expected
choices=json.loads((ROOT/'config/final_method_choices.json').read_text());rows=[];cost=[];out=ROOT/'results/skt_predictions';out.mkdir(exist_ok=True)
for robot,ch in choices.items():
 pool=dict(np.load(ROOT/'results/skt_aligned'/robot/'pool.npz'));orig=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));X=pool['X'];Y=pool['Y'];hp=ch['predictor'];sets=[('retrospective',BASE/'data'/robot/'test.npz')]
 if robot=='MELFA':sets.append(('new_records',ROOT/'data/MELFA_holdout.npz'))
 for stage,path in sets:
  d=dict(np.load(path));V=(d['X']-pool['mu'])/pool['sd']
  for name in sorted(expected):
   if not name.startswith('results/skt_aligned/'+robot+'/'):continue
   method,seed,m=Path(name).stem.rsplit('_',2);seed=int(seed);m=int(m);ix=np.load(ROOT/name);start=time.perf_counter();pred,_=core.fit_predict(core.kernel(X[ix],X[ix],hp['scale']),core.kernel(V,X[ix],hp['scale']),Y[ix],hp['lambda_mean']);physical=pred*pool['ys']+pool['ym'];tag=f'{robot}_{stage}_{method}_{seed}_{m}';p=out/(tag+'.npz');assert not p.exists();np.savez_compressed(p,pred_physical=physical,indices=ix);cost.append(dict(tag=tag,fit_predict_seconds=time.perf_counter()-start))
   for g,r in enumerate(d['names']):
    a=d['group']==g;e=(physical[a]-d['Y'][a])**2;rows.append(dict(robot=robot,stage=stage,method=method,seed=seed,budget=m,record=str(r),rows=int(a.sum()),nmse=float(np.mean(e/orig['ys']**2)),rmse_Nm=float(np.sqrt(e.mean())),prediction=p.relative_to(ROOT).as_posix()))
 print(robot,'aligned evaluated',flush=True)
(ROOT/'results/skt_evaluation_rows.json').write_text(json.dumps(rows,indent=2));(ROOT/'results/skt_evaluation_cost.json').write_text(json.dumps(cost,indent=2))
