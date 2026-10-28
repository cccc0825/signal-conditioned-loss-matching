"""Post-freeze descriptive allocation and timer summaries; no method search."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
d=pd.DataFrame(json.loads((ROOT/'results/selection_times.json').read_text()))
d.groupby(['robot','method'],as_index=False)[['selection_seconds','kernel_build_seconds']].mean().to_csv(ROOT/'analysis/selection_cost_summary.csv',index=False)
rows=[]
for robot in ['PANDA','MELFA']:
 t=dict(np.load(ROOT/'results'/f'{robot}_target.npz'));bins=t['bins']
 for p in sorted((ROOT/'results/selections'/robot).glob('*.npy')):
  method,seed=p.stem.rsplit('_',1)
  if int(seed) in [7320,7321,7322]:continue
  ix=np.load(p)
  for m in [64,128,256]:
   for b in np.unique(bins):
    rows.append(dict(robot=robot,method=method,seed=int(seed),budget=m,bin=int(b),candidate_fraction=float(np.mean(bins==b)),target_mass=float(t['weights'][bins==b].sum()),selected_fraction=float(np.mean(bins[ix[:m]]==b))))
pd.DataFrame(rows).to_csv(ROOT/'analysis/condition_allocation.csv',index=False)
