"""Three distinct evidence roles; every plotted quantity comes from frozen runs."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'analysis';F=ROOT/'figures';F.mkdir(exist_ok=True)
paths=['signal_difficulty.csv','summary.csv','new_record_nmse.csv','influence.csv']
contract=dict(question='Where does signal-conditioned weighting help finite robot dictionaries?',panels={'a':'Development mechanism: all training records and all energy quartiles','b':'Frozen new-record budget response with a stronger baseline retained','c':'Frozen new-record component differences and recording influence'},archetype='complementary evidence panels',backend='Python',size_inches=[6.875,1.95],new_experiments=0,statistics='Record means; stochastic selectors averaged over ten seeds within record; panel c whiskers are leave-one-record-out ranges',hashes={p:hashlib.sha256((A/p).read_bytes()).hexdigest() for p in paths})
(A/'figure_contract.json').write_text(json.dumps(contract,indent=2))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False})
fig,axs=plt.subplots(1,3,figsize=(6.875,1.95),gridspec_kw={'width_ratios':[1,1,1.23]});fig.subplots_adjust(left=.068,right=.991,bottom=.24,top=.82,wspace=.72)
blue='#0072B2';orange='#D55E00';gray='#666666';purple='#8E6C8A'
d=pd.read_csv(A/paths[0]);ax=axs[0]
for robot,color,marker in [('PANDA',blue,'o'),('MELFA',orange,'s')]:
 s=d[d.robot==robot]
 for name,g in s.groupby('record'):ax.scatter(g.quartile,g.nmse,s=7,color=color,alpha=.25,linewidths=0)
 v=s.groupby('quartile').nmse.mean();ax.plot(v.index,v.values,color=color,marker=marker,markersize=3,lw=1.2,label='Panda' if robot=='PANDA' else 'MELFA')
ax.set_title('(a) Signal difficulty',loc='left',fontweight='bold',pad=8);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Acceleration-energy quartile');ax.set_ylabel('Cross-record pilot NMSE');ax.set_ylim(bottom=0);ax.legend(loc='upper left',handlelength=1,handletextpad=.4)
s=pd.read_csv(A/paths[1]);s=s[(s.robot=='MELFA')&(s.stage=='new_records')&(s.variant=='shared')];ax=axs[1]
for method,label,color,marker in [('random','Random',gray,'o'),('full','Full loss',orange,'^'),('kcenter','k-center',purple,'s'),('signal_task','Signal rule',blue,'D')]:
 v=s[s.method==method].sort_values('budget');ax.plot(v.budget,v.nmse,color=color,marker=marker,markersize=3,lw=1.2,label=label)
ax.set_title('(b) Held-out record error',loc='left',fontweight='bold',pad=8);ax.set_xticks([64,128,256]);ax.set_xlabel('Dictionary size');ax.set_ylabel('Mean NMSE');ax.legend(loc='upper right',handlelength=1,handletextpad=.3,labelspacing=.2)
p=pd.read_csv(A/paths[2],index_col=0);inf=pd.read_csv(A/paths[3]);ax=axs[2]
methods=['weighted_random','residual_only','random_bins','signal_static'];labels=['No matching','Point residual','Permuted bins','Fixed weights'];values=[]
for y,m in enumerate(methods):
 v=100*(p[m].mean()/p.signal_task.mean()-1);g=inf[inf.comparator==m].comparator_percent_above_rule;ax.plot([g.min(),g.max()],[y,y],color=gray,lw=1);ax.scatter([v],[y],s=23,color=blue,zorder=3);values.append(dict(ablation=m,percent_above_rule=v,loo_min=float(g.min()),loo_max=float(g.max())))
ax.axvline(0,color='#555555',ls='--',lw=.8);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlabel('NMSE change vs rule (%)');ax.set_title('(c) Ablation sensitivity',loc='left',fontweight='bold',pad=8);ax.set_ylim(3.5,-.5)
fig.savefig(F/'signal_evidence.pdf');fig.savefig(F/'signal_evidence.svg');fig.savefig(F/'signal_evidence.png',dpi=600);plt.close(fig)
(A/'ablation_plot_values.json').write_text(json.dumps(values,indent=2))
print('Exported source-grounded three-panel figure.')
