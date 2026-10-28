"""Record-first aggregation and traceable paper numbers from all frozen outcomes."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'analysis';out.mkdir(exist_ok=True)
D=pd.DataFrame(json.loads((ROOT/'results/evaluation_rows.json').read_text()));A=pd.DataFrame(json.loads((ROOT/'results/skt_evaluation_rows.json').read_text()))
R=D.groupby(['robot','stage','method','budget','variant','record'],as_index=False)[['nmse','rmse_Nm']].mean();S=R.groupby(['robot','stage','method','budget','variant'],as_index=False)[['nmse','rmse_Nm']].mean();R.to_csv(out/'record_means.csv',index=False);S.to_csv(out/'summary.csv',index=False)
AR=A.groupby(['robot','stage','method','budget','record'],as_index=False)[['nmse','rmse_Nm']].mean();AS=AR.groupby(['robot','stage','method','budget'],as_index=False)[['nmse','rmse_Nm']].mean();AR.to_csv(out/'skt_record_means.csv',index=False);AS.to_csv(out/'skt_summary.csv',index=False)
P=R[(R.robot=='MELFA')&(R.stage=='new_records')&(R.budget==256)&(R.variant=='shared')].pivot(index='record',columns='method',values='nmse');P.to_csv(out/'new_record_nmse.csv');influence=[]
for other in P.columns:
 delta=P.signal_task-P[other]
 for record in P.index:
  keep=P.index!=record;influence.append(dict(comparator=other,omitted=record,full_delta=float(delta.mean()),loo_delta=float(delta[keep].mean()),comparator_percent_above_rule=float(100*(P[other][keep].mean()/P.signal_task[keep].mean()-1))))
pd.DataFrame(influence).to_csv(out/'influence.csv',index=False)
# All four energy quartiles in all training records, pilot residuals only.
signals=[]
for robot in ['PANDA','MELFA']:
 tr=dict(np.load(ROOT/'data'/f'{robot}_train.npz'));err=np.load(ROOT/'results'/f'{robot}_pilot.npz')['error'];cut=np.quantile(tr['Z'][:,0],[.25,.5,.75]);b=np.digitize(tr['Z'][:,0],cut)
 for g,name in enumerate(tr['names']):
  for q in range(4):
   mask=(tr['group']==g)&(b==q);signals.append(dict(robot=robot,record=str(name),quartile=q+1,n=int(mask.sum()),nmse=float(err[mask].mean()) if mask.any() else None))
pd.DataFrame(signals).to_csv(out/'signal_difficulty.csv',index=False)
def val(robot,stage,method,metric='nmse',variant='shared',budget=256):return float(S[(S.robot==robot)&(S.stage==stage)&(S.method==method)&(S.budget==budget)&(S.variant==variant)][metric].iloc[0])
metrics={'PrimaryRule':val('MELFA','new_records','signal_task'),'PrimaryRandom':val('MELFA','new_records','random'),'PrimaryImprovement':100*(1-val('MELFA','new_records','signal_task')/val('MELFA','new_records','random')),'PrimaryRmse':val('MELFA','new_records','signal_task','rmse_Nm'),'RandomRmse':val('MELFA','new_records','random','rmse_Nm')}
(rootpaper:=ROOT/'paper/numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+(f'{v:.1f}' if k=='PrimaryImprovement' else f'{v:.4f}')+'}' for k,v in metrics.items())+'\n')
labels={'random':'Random','kcenter':'$k$-center','variance':'GP variance','gemss_mo':'GEMSS-MO','centered':'Centered loss','full':'Full loss','signal_task':'Signal-conditioned'}
lines=[]
for method,label in labels.items():
 vals=[val('PANDA','retrospective',method),val('MELFA','retrospective',method),val('MELFA','new_records',method),val('MELFA','new_records',method,'rmse_Nm'),val('MELFA','new_records',method,variant='retuned')]
 lines.append(label+' & '+' & '.join(f'{x:.4f}' for x in vals)+r' \\')
(ROOT/'paper/main_rows.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for method in ['random','centered','full','skt','signal_task']:
 a=AS[(AS.robot=='MELFA')&(AS.stage=='new_records')&(AS.method==method)&(AS.budget==256)].iloc[0];label=labels.get(method,'SKT (direct KT)');lines.append(label+f" & {a.nmse:.4f} & {a.rmse_Nm:.4f}"+r' \\')
(ROOT/'paper/skt_rows.tex').write_text('\n'.join(lines)+'\n');(out/'paper_values.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
