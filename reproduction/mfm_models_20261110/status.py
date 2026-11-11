"""On-demand status/report; never monitors or launches a process."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent

def main():
    tables=[];persons=[];jobs=[]
    for directory in ('results','results_spectral_inputs'):
        root=HERE/directory
        if (root/'comparison.csv').exists():
            a=pd.read_csv(root/'comparison.csv');a['output_dir']=directory;tables.append(a)
        if (root/'by_person.csv').exists():persons.append(pd.read_csv(root/'by_person.csv'))
        for path in (root/'jobs').glob('*.json'):
            r=json.loads(path.read_text());jobs.append(dict(family=r['family'],tag=r['tag'],status=r['status'],
                seconds=r.get('seconds'),failures=len(r.get('failures',r.get('failed',{})))))
    if tables:
        df=pd.concat(tables).drop_duplicates('name',keep='first').sort_values('oof_cell',ascending=False)
        for name,label in [('historical_lag4_50__balance','delta_vs_historical_balanced_pp'),('lag4_matched_lr50__balance','delta_vs_matched_balanced_pp')]:
            base=df.loc[df.name==name,'oof_cell']
            df[label]=100*(df.oof_cell-float(base.iloc[0])) if len(base) else np.nan
        df.to_csv(HERE/'COMBINED_COMPARISON.csv',index=False)
        print(df.head(20).to_string(index=False))
        if persons:
            p=pd.concat(persons).drop_duplicates(['name','tag'],keep='first')
            p.to_csv(HERE/'COMBINED_BY_PERSON.csv',index=False)
    j=pd.DataFrame(jobs);j.to_csv(HERE/'JOB_STATUS.csv',index=False)
    print('\nCompleted/failed outer jobs:',len(jobs),'of77 expected; failed policies:',sum(r['failures'] for r in jobs))
    if len(j):print(j.groupby(['family','status']).size().to_string())

if __name__=='__main__':main()
