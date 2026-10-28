"""Run in isolated SKT venv; no test data access."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'skt_baseline'))
from skt_wrapper import select
rows=[]
for robot in ['PANDA','MELFA']:
 p=ROOT/'results/skt_aligned'/robot;c=json.loads((p/'skt_call.json').read_text());d=dict(np.load(p/'skt_input.npz'))
 for m in c['budgets']:
  for seed in c['seeds']:
   idx,identity=select(d['X'],d['Y'],c['scale'],m,seed);np.save(p/f'skt_{seed}_{m}.npy',idx);rows.append(dict(robot=robot,**identity))
 print(robot,'SKT finished',flush=True)
(ROOT/'results/skt_aligned/author_runs.json').write_text(json.dumps(rows,indent=2))
