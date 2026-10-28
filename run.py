#!/usr/bin/env python3
"""Portable replay of the published dictionaries and record-level evaluation."""
from pathlib import Path, PurePosixPath
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, zipfile
ROOT = Path(__file__).resolve().parent

def sha(data): return hashlib.sha256(data).hexdigest()

def verify():
    manifest=json.loads((ROOT/'config/public_manifest.json').read_text())
    with zipfile.ZipFile(ROOT/'artifacts/frozen_inputs.zip') as archive:
        for name, digest in manifest['files'].items():
            data=archive.read(name) if name in archive.namelist() else (ROOT/name).read_bytes()
            if sha(data)!=digest: raise ValueError('Integrity mismatch: '+name)
    print('Verified',len(manifest['files']),'release input files',flush=True)
    return manifest

def create_run(output):
    out=Path(output).expanduser().resolve() if output else ROOT/'runs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if out.exists(): raise ValueError('Output must be a new directory: '+str(out))
    out.mkdir(parents=True)
    for folder in ['code','config','skt_baseline']:shutil.copytree(ROOT/folder,out/folder,ignore=shutil.ignore_patterns('__pycache__'))
    with zipfile.ZipFile(ROOT/'artifacts/frozen_inputs.zip') as archive:
        for info in archive.infolist():
            name=PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe archive path')
            archive.extract(info,out)
    for folder in ['paper','analysis','figures','logs']:(out/folder).mkdir(exist_ok=True)
    return out

def run_script(out,name,python=None):
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'logs/matplotlib'))
    print('Running',name,flush=True)
    with (out/'logs'/(Path(name).stem+'.log')).open('w') as log:
        subprocess.run([python or sys.executable,str(out/'code'/name)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def compare_metrics(out):
    import numpy as np
    report={'status':'PASS','metric_rows':0,'maximum_absolute_difference':0.0,'files':{}}
    keys=['robot','stage','method','seed','budget','variant','record']
    for name in ['evaluation_rows.json','skt_evaluation_rows.json']:
        expected=json.loads((out/'expected'/name).read_text());actual=json.loads((out/'results'/name).read_text())
        key=lambda r:tuple(r.get(k) for k in keys)
        E={key(r):r for r in expected};A={key(r):r for r in actual}
        assert len(E)==len(expected) and len(A)==len(actual) and set(E)==set(A),name+' row inventory mismatch'
        maximum=0.0
        for k,e in E.items():
            a=A[k];assert e['rows']==a['rows']
            for metric in ['nmse','rmse_Nm','joint_nmse']:
                if metric not in e:continue
                error=float(np.max(np.abs(np.asarray(e[metric])-np.asarray(a[metric]))));maximum=max(maximum,error)
                np.testing.assert_allclose(a[metric],e[metric],rtol=1e-10,atol=1e-10,err_msg=str((k,metric)))
        report['files'][name]={'rows':len(A),'maximum_absolute_difference':maximum};report['metric_rows']+=len(A);report['maximum_absolute_difference']=max(maximum,report['maximum_absolute_difference'])
    report['prediction_files']=sum(len(list((out/'results'/name).glob('*.npz'))) for name in ['predictions','skt_predictions'])
    assert report['prediction_files']==771
    report['python']=sys.version;report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

def compare_selections(out,include_skt):
    import numpy as np
    with zipfile.ZipFile(ROOT/'artifacts/frozen_inputs.zip') as archive:
        import io
        selected=[n for n in archive.namelist() if n.endswith('.npy') and n.startswith(('results/selections/','results/skt_aligned/')) and (include_skt or not Path(n).name.startswith('skt_'))]
        for name in selected:np.testing.assert_array_equal(np.load(out/name),np.load(io.BytesIO(archive.read(name))),err_msg=name)
    report={'status':'PASS','dictionary_files_compared':len(selected),'includes_author_skt_rebuild':include_skt,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (out/'selector_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','reproduce','selectors','extract']);p.add_argument('--output');p.add_argument('--skt-python',help='Optional Python with the archived author goodpoints installed');p.add_argument('--development',action='store_true',help='Replay the 24 validation-based development configurations before rebuilding selectors')
    a=p.parse_args();verify()
    if a.command=='verify':return
    out=create_run(a.output)
    if a.command=='reproduce':
        for name in ['test_method.py','evaluate.py','evaluate_skt.py','analyze.py','analyze_allocation_cost.py','plot_results.py']:run_script(out,name)
        compare_metrics(out)
    elif a.command=='selectors':
        if a.development:
            for name in ['develop.py','develop_weighted.py']:run_script(out,name)
        for name in ['build_evaluation.py','prepare_skt_comparison.py']:run_script(out,name)
        # Keep the virtual-environment entry point; resolving its symlink can
        # bypass the environment where the author baseline was installed.
        if a.skt_python:run_script(out,'run_skt_aligned.py',str(Path(a.skt_python).expanduser().absolute()))
        compare_selections(out,bool(a.skt_python))
    print('Outputs:',out)
if __name__=='__main__':main()
