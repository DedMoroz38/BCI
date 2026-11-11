"""Post-run paired descriptive comparisons; no model selection or refitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent

def main():
    comparison=pd.read_csv(HERE/'comparison.csv').set_index('model')
    people=pd.read_csv(HERE/'per_person.csv')
    rows=[];detail=[]
    for candidate in comparison.index:
        if not candidate.startswith('mssa8_mix25_nuclear'):continue
        balanced=candidate.endswith('__balance');suffix='__balance' if balanced else ''
        c=people[people.model==candidate].set_index('person')
        for ref in ('mssa8_mix25_lr50','lag4_nuclear001_50','lag4_nuclear005_50'):
            baseline=ref+suffix
            if baseline not in comparison.index:continue
            b=people[people.model==baseline].set_index('person');delta=c.first_ba-b.first_ba
            rows.append(dict(candidate=candidate,baseline=baseline,first_session_ba=float(comparison.loc[candidate,'first_session_ba']),
                             baseline_ba=float(comparison.loc[baseline,'first_session_ba']),delta_pp=float(100*delta.mean()),
                             people_better=int((delta>1e-12).sum()),people_tied=int((abs(delta)<=1e-12).sum()),people_worse=int((delta < -1e-12).sum()),
                             worst3_delta_pp=float(100*(comparison.loc[candidate,'worst3']-comparison.loc[baseline,'worst3'])),
                             first_nll_delta=float(comparison.loc[candidate,'first_nll']-comparison.loc[baseline,'first_nll'])))
            for person,value in delta.items():detail.append(dict(candidate=candidate,baseline=baseline,person=person,delta_pp=100*value))
    df=pd.DataFrame(rows);df.to_csv(HERE/'paired_comparisons.csv',index=False)
    pd.DataFrame(detail).to_csv(HERE/'paired_per_person.csv',index=False)
    status=json.loads((HERE/'STATUS.json').read_text())
    (HERE/'PAIRED_REPORT.md').write_text('# Paired CPU synergy comparisons\n\nDifferences in percentage points; person-level metrics averaged across the two repeats before comparison.\n\n'+df.to_string(index=False)+'\n\n'+f"Runtime {status['seconds']:.1f}s; failed folds {status['failed_folds']}; failed head configurations {status['failed_head_configs']}.\n")
    print(df.to_string(index=False))

if __name__=='__main__':main()
