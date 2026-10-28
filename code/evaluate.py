"""Evaluate frozen methods without retuning; save every per-record prediction and metric."""
from pathlib import Path
import json,time,hashlib,datetime,sys
import numpy as np
import legacy_core as core
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reference'
def main():
 fr=json.loads((ROOT/'config/public_manifest.json').read_text())
 for n,h in fr['files'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 expected={n for n in fr['files'] if n.startswith('results/selections/') and n.endswith('.npy')}
 actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'results/selections').rglob('*.npy')}
 assert actual==expected,'Selection directory differs from frozen inventory'
 rows=[];cost=[];start=time.time();choices=json.loads((ROOT/'config/final_method_choices.json').read_text());secondary=json.loads((ROOT/'config/secondary_predictor_choices.json').read_text());pdir=ROOT/'results/predictions';pdir.mkdir(exist_ok=True)
 for robot,ch in choices.items():
  tr=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));X=(tr['X']-tr['mu'])/tr['sd'];Y=(tr['Y']-tr['ym'])/tr['ys'];sets=[('retrospective',BASE/'data'/robot/'test.npz')]
  if robot=='MELFA':sets.append(('new_records',ROOT/'data/MELFA_holdout.npz'))
  for stage,path in sets:
   d=dict(np.load(path));V=(d['X']-tr['mu'])/tr['sd'];VY=(d['Y']-tr['ym'])/tr['ys']
   for f in [ROOT/n for n in sorted(expected) if n.startswith('results/selections/'+robot+'/')]:
    method,seed=f.stem.rsplit('_',1);seed=int(seed)
    if seed in [7320,7321,7322]:continue
    allix=np.load(f)
    for m in [64,128,256]:
     ix=allix[:m]
     for variant in (['shared','retuned'] if m==256 else ['shared']):
      hp=ch['predictor'] if variant=='shared' else secondary[f'{robot}_{method}'];t=time.perf_counter();P,C=core.fit_predict(core.kernel(X[ix],X[ix],hp['scale']),core.kernel(V,X[ix],hp['scale']),Y[ix],hp['lambda_mean']);elapsed=time.perf_counter()-t;assert np.isfinite(P).all();tag=f'{robot}_{stage}_{method}_{seed}_{m}_{variant}';out=pdir/(tag+'.npz');assert not out.exists(),tag
      np.savez_compressed(out,pred_standardized=P,indices=ix,coefficients=C)
      for g,name in enumerate(d['names']):
       mask=d['group']==g;e=(P[mask]-VY[mask])**2;physical=e*tr['ys']**2
       rows.append(dict(robot=robot,stage=stage,method=method,seed=seed,budget=m,variant=variant,record=str(name),rows=int(mask.sum()),nmse=float(e.mean()),rmse_Nm=float(np.sqrt(physical.mean())),joint_nmse=e.mean(0).tolist(),prediction=out.relative_to(ROOT).as_posix(),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
      cost.append(dict(tag=tag,fit_predict_seconds=elapsed))
   print('EVALUATED',robot,stage,len(rows),flush=True)
 (ROOT/'results/evaluation_rows.json').write_text(json.dumps(rows,indent=2));(ROOT/'results/evaluation_cost.json').write_text(json.dumps(cost,indent=2));(ROOT/'results/evaluation_execution.json').write_text(json.dumps(dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),predictions=len(cost),metric_rows=len(rows),seconds=time.time()-start,python=sys.version,freeze_sha256=hashlib.sha256((ROOT/'config/public_manifest.json').read_bytes()).hexdigest()),indent=2))
if __name__=='__main__':main()
