"""Candidate-only CLI, using the isolated author dependency environment."""
import argparse
import json
from pathlib import Path
import numpy as np
from skt_wrapper import select
p=argparse.ArgumentParser()
p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--scale',required=True,type=float);p.add_argument('--m',required=True,type=int)
p.add_argument('--seed',required=True,type=int)
a=p.parse_args()
with np.load(a.input,allow_pickle=False) as z:
    idx,identity=select(z['X'],z['Y'],a.scale,a.m,a.seed)
np.savez(a.output,indices=idx)
Path(a.output+'.json').write_text(json.dumps(identity,indent=2))
