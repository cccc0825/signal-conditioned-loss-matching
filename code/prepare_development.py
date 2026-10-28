"""Only historical training/validation members; source index gap-aware signal windows."""
from pathlib import Path
import numpy as np,json,zipfile,hashlib
from scipy.ndimage import uniform_filter1d
from source_reader import read
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'reference';ARCHIVE=ROOT/'raw/LIP4RID.zip'
def features(x,positions,vscale,ascale,window=31):
 j=len(vscale);e=np.column_stack([np.mean(x[:,j:2*j]**2/vscale,axis=1),np.mean(x[:,2*j:]**2/ascale,axis=1)])
 gaps=np.flatnonzero(np.diff(positions)!=1)+1;cuts=np.r_[0,gaps,len(x)];s=np.empty_like(e)
 for lo,hi in zip(cuts[:-1],cuts[1:]):s[lo:hi]=uniform_filter1d(e[lo:hi],size=window,axis=0,mode='nearest')
 return np.column_stack([np.log1p(s[:,1]),np.log((s[:,1]+1e-6)/(s[:,0]+1e-6))])
def main():
 man=json.loads((BASE/'research/lip_data_manifest.json').read_text());audit=[]
 with zipfile.ZipFile(ARCHIVE) as z:
  for robot in ['PANDA','MELFA']:
   rec={}
   for r in man['records']:
    if r['robot']!=robot or r['split'] not in ['train','validation']:continue
    b=z.read(r['source']);assert hashlib.sha256(b).hexdigest()==r['sha256'];df=read(b);j=r['dof'];x=df[r['columns'][:3*j]].to_numpy(float);y=df[r['columns'][3*j:]].to_numpy(float);pos=np.asarray(df.index,int)
    assert len(pos)==len(x) and np.all(np.diff(pos)>0)
    rec[r['id']]=(x,y,pos)
    audit.append(dict(record=r['id'],source=r['source'],source_sha256=r['sha256'],index_gaps=int(np.sum(np.diff(pos)!=1)),timestamp='observed t' if 't' in df else 'no timestamp; derivative energy descriptors only',rows=len(x)))
   tr=dict(np.load(BASE/'data'/robot/'train.npz'));allx=np.concatenate([rec[str(n)][0] for n in tr['names']]);j=tr['Y'].shape[1];vs=np.maximum(np.mean(allx[:,j:2*j]**2,axis=0),1e-12);acs=np.maximum(np.mean(allx[:,2*j:]**2,axis=0),1e-12)
   for split in ['train','validation']:
    d=dict(np.load(BASE/'data'/robot/(split+'.npz')));xs=[];ys=[];zz=[]
    for n in d['names']:
     x,y,pos=rec[str(n)];xs.append(x[::10]);ys.append(y[::10]);zz.append(features(x,pos,vs,acs)[::10])
    assert np.array_equal(np.concatenate(xs),d['X']) and np.array_equal(np.concatenate(ys),d['Y'])
    np.savez_compressed(ROOT/'data'/f'{robot}_{split}.npz',**d,Z=np.concatenate(zz))
   (ROOT/'data'/f'{robot}_feature_config.json').write_text(json.dumps(dict(velocity_scale=vs.tolist(),acceleration_scale=acs.tolist(),window_rows=31,epsilon=1e-6,normalization='full historical training recordings only',units='dimensionless standardized derivative energies; no temporal Hz assigned',gap_handling='No smoothing across nonconsecutive source dataframe indices',state_predictor_unchanged=True),indent=2))
 (ROOT/'research/development_source_audit.json').write_text(json.dumps(audit,indent=2));print('Prepared original train/validation pools with gap-aware signal descriptors; holdout untouched.')
if __name__=='__main__':main()
