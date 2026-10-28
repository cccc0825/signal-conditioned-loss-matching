"""Convert checksum-verified public LIP4RID observations, not supplied model outputs."""
import argparse,hashlib,io,json,pickle,re,zipfile,importlib
from pathlib import Path
import numpy as np
import pandas as pd

def compatible_new_block(values,placement,ndim,**kwargs):
    from pandas.core.internals.blocks import new_block
    from pandas._libs.internals import BlockPlacement
    return new_block(values,placement=BlockPlacement(placement),ndim=ndim,**kwargs)

class RestrictedUnpickler(pickle.Unpickler):
    allowed={('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),('functools','partial'),('pandas.core.internals.blocks','new_block'),('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('builtins','slice'),('pandas.core.indexes.base','_new_Index'),('pandas.core.indexes.base','Index'),('pandas.core.indexes.numeric','Int64Index'),('pandas.core.indexes.range','RangeIndex'),('numpy.core.multiarray','scalar')}
    def find_class(self,module,name):
        if (module,name) not in self.allowed:raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
        if (module,name)==('pandas.core.indexes.numeric','Int64Index'):return pd.Index
        if (module,name)==('pandas.core.internals.blocks','new_block'):return compatible_new_block
        return getattr(importlib.import_module(module),name)

def read(b):return RestrictedUnpickler(io.BytesIO(b)).load()

def main():
    a=argparse.ArgumentParser();a.add_argument('archive');a.add_argument('destination');a=a.parse_args();p=Path(a.archive);out=Path(a.destination);out.mkdir(parents=True,exist_ok=True)
    md5=hashlib.md5(p.read_bytes()).hexdigest();assert md5=='d3e29fbd280fc4cf08d008eb50559338'
    z=zipfile.ZipFile(p);records=[];excluded=[]
    for name in z.namelist():
        if '/Robots/' not in name or '/Experiments/' not in name or not name.endswith('.pkl'):continue
        robot='PANDA' if '/PANDA/' in name else 'MELFA';seed=int(re.search(r'seed_?(\d+)',name).group(1))
        if robot=='PANDA':
            if not name.endswith('_as_filtered_fcut4.0.pkl'):excluded.append({'file':name,'reason':'unfiltered duplicate or computed dynamics; not used'});continue
            split='test' if '_sin_100_' in name else ('validation' if seed in [16,18] else 'train');d=7;pref=''
        else:
            is_test='pos_range_1.4' in name
            # Follow README enumerated membership; extra files are not part of declared benchmark.
            valid=[1,2,3,4,6,7,8,9,11] if is_test else [0,1,2,3,4,7,8,9,10]
            if seed not in valid:excluded.append({'file':name,'reason':'not in dataset README declared benchmark seed list; decided before modeling'});continue
            split='test' if is_test else ('validation' if seed in [9,10] else 'train');d=6;pref='_filt'
        b=z.read(name); df=read(b)
        cols=[f'{kind}{pref}_{j}' for kind in ['q','dq','ddq'] for j in range(1,d+1)]; ycols=[f'tau{pref}_{j}' for j in range(1,d+1)]
        X=df[cols].to_numpy(float);Y=df[ycols].to_numpy(float)
        assert np.isfinite(X).all() and np.isfinite(Y).all(),name
        # Preserve all supplied samples; model candidates are decimated later.
        key=f'{robot}_{split}_{seed}'; t=df['t'].to_numpy() if 't' in df else np.arange(len(df))
        np.savez_compressed(out/(key+'.npz'),X=X,Y=Y,t=t)
        rec=dict(id=key,robot=robot,split=split,seed=seed,rows=len(df),dof=d,source=name,sha256=hashlib.sha256(b).hexdigest(),columns=cols+ycols,finite=True,exact_duplicate_inputs=int(len(X)-len(np.unique(X,axis=0))))
        records.append(rec); print(key,X.shape,flush=True)
    manifest={'source':'https://zenodo.org/records/12516500','doi':'10.5281/zenodo.12516500','license':'CC-BY-SA-4.0','archive_md5':md5,'records':records,'excluded':excluded,'note':'Source README benchmark seed membership is used; paper, archive and README counts differ. No author model/result files are read.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
