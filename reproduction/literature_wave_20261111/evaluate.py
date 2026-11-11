"""Person/repeat aggregation; repeated model predictions are never ensembled."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core import META,score,subset,streams,HERE

def observations(scheme,tag,name,m,p):
    rows=[]
    for person in np.unique(m['subject']):
        ids=np.flatnonzero(m['subject']==person)
        first=ids[m['session'][ids]=='1'];first=first[np.lexsort((m['onset'][first],m['run'][first]))]
        row=dict(scheme=scheme,fold=tag,model=name,person=person,first_windows=len(first),all_windows=len(ids),
                 first_ba=score(subset(m,first),p[first])['cell'],first_nll=score(subset(m,first),p[first])['nll'],
                 all_ba=score(subset(m,ids),p[ids])['cell'],all_cells=len(set(zip(m['session'][ids],m['context'][ids]))))
        for label,part in [('first20',first[:20]),('middle21_60',first[20:60]),('tail61',first[60:])]:
            counts=np.bincount(m['y'][part],minlength=3);pred=p[part].argmax(1)
            # Macro recall over3classes; missing-class support is explicitly NaN.
            ba=float(np.mean([np.mean(pred[m['y'][part]==c]==c) for c in range(3)])) if np.all(counts>0) else np.nan
            row[label+'_ba']=ba;row[label+'_n']=len(part);row[label+'_counts']=json.dumps(counts.tolist())
            row[label+'_contexts']=json.dumps({str(c):int(np.sum(m['context'][part]==c)) for c in np.unique(m['context'][part])})
        rows.append(row)
    return rows

def summarize(inputs,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);obs=[];seen={}
    expected={'loso':10,'pair8_2':10};people_counts={'loso':1,'pair8_2':2}
    for scheme,root in inputs:
        for f in sorted((Path(root)/'predictions').glob('*/*.npz')):
            tag=f.parent.name
            if tag=='later_last':continue
            with np.load(f) as z:m={k:z[k] for k in META};p=z['proba']
            key=(scheme,f.stem,tag)
            if key in seen:continue
            seen[key]=1;obs+=observations(scheme,tag,f.stem,m,p)
    if not obs:return
    detail=pd.DataFrame(obs);detail.to_csv(out/'per_person_repeat.csv',index=False);rows=[];per=[];incomplete=[]
    for (scheme,name),df in detail.groupby(['scheme','model']):
        counts=df.groupby('person').size()
        complete=df.fold.nunique()==expected[scheme] and len(counts)==10 and counts.eq(people_counts[scheme]).all()
        if not complete:
            incomplete.append(dict(scheme=scheme,model=name,folds=df.fold.nunique(),people=len(counts)));continue
        agg=df.groupby('person')[['first_ba','first_nll','all_ba','first20_ba','middle21_60_ba','tail61_ba']].mean()
        for person,r in agg.iterrows():per.append(dict(scheme=scheme,model=name,person=person,**r.to_dict()))
        spread=df.groupby('person').first_ba.std(ddof=0).mean()
        rows.append(dict(scheme=scheme,model=name,first_session_ba=float(agg.first_ba.mean()),
                         worst3=float(np.sort(agg.first_ba)[:3].mean()),median_person=float(agg.first_ba.median()),
                         all_session_cell_ba=float(np.average(df.all_ba,weights=df.all_cells)),
                         first_nll=float(agg.first_nll.mean()),first20_ba=float(agg.first20_ba.mean()),
                         middle21_60_ba=float(agg.middle21_60_ba.mean()),tail61_ba=float(agg.tail61_ba.mean()),
                         mean_repeat_sd=float(spread),folds=int(df.fold.nunique())))
    result=pd.DataFrame(rows)
    if len(result):result=result.sort_values(['scheme','first_session_ba'],ascending=[True,False])
    result.to_csv(out/'comparison.csv',index=False);pd.DataFrame(per).to_csv(out/'per_person.csv',index=False)
    pd.DataFrame(incomplete).to_csv(out/'incomplete.csv',index=False)
    text='# Approved custom validation\n\nPrimary:first session, equal people;8/2 repeats average metrics,not probabilities.\n'
    text+='LOSO cached results and8/2 retraining are separate scenarios. Stream macro-BA bins1–20,21–60,61+ retain natural session state.\n\n'
    text+=result.to_string(index=False) if len(result) else 'No complete model yet.'
    (out/'REPORT.md').write_text(text+'\n')

if __name__=='__main__':
    summarize([('loso',HERE/'results_loso'),('pair8_2',HERE/'results_pair_cpu'),('pair8_2',HERE/'results_pair_gpu')],HERE/'reports')
