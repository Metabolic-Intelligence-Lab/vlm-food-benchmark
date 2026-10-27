"""Verbosity of predicted food names vs recognition.

Mean words per predicted name and items per image for every model at its
best-F1 configuration (eps=0.5; v1 clean + v2 r1), with the Spearman
correlation between verbosity and F1.

Usage: python analysis/17_verbosity.py
"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'analysis')
from evaluation import load_run_predictions, load_ground_truth
import importlib.util
spec=importlib.util.spec_from_file_location('m12','analysis/12_eval_all_v2.py'); m12=importlib.util.module_from_spec(spec); spec.loader.exec_module(m12)
gt=load_ground_truth(); gtlen=np.mean([len(n.split()) for v in gt.values() for n,_ in v])
clean=json.load(open('analysis/outputs/metrics_clean.json'))
best={}
for v in clean.values():
    if v['epsilon']==0.5 and not v['model'].startswith('gpt-4o-mini'):
        if v['f1']>best.get(v['model'],(0,))[0]: best[v['model']]=(v['f1'],v['temperature'],v['precision'],v['recall'])
rows=[]
for m,(f1,t,p,r) in best.items():
    preds=load_run_predictions(m,f"{t:.1f}",empty_raw_as_empty=True)
    names=[n for pl,_ in preds.values() for n,_ in pl]
    rows.append((m,f1,p,r,np.mean([len(n.split()) for n in names]),len(names)/200))
v2=pd.read_csv('analysis/outputs/metrics_v2_all.csv'); v2=v2[(v2.epsilon==0.5)&(v2.run=='r1')]
runs=m12.load_all_runs()
for m,g in v2.groupby('model'):
    if m in best: continue
    b=g.loc[g.f1.idxmax()]
    by=runs[(m,f"{b.temperature:.1f}",'r1')]
    names=[n for pl in by.values() for n,_ in pl]
    rows.append((m,b.f1,b.precision,b.recall,np.mean([len(n.split()) for n in names]),len(names)/200))
df=pd.DataFrame(rows,columns=['model','f1','prec','rec','words_per_name','items_per_img']).sort_values('words_per_name')
print(df.round(3).to_string(index=False)); print('GT words/name',round(gtlen,2),' GT items/img 4.35')
print('Spearman words vs F1 (all):',spearmanr(df.words_per_name,df.f1))
u=df[df.items_per_img>2]; print('Spearman (models with >2 items/img):',spearmanr(u.words_per_name,u.f1), len(u))
