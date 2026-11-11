"""Rescore saved top-model forecasts, with no fitting or altered stream state."""
from pathlib import Path
import shutil
import numpy as np
from core import *
from evaluate import summarize

def main():
    out=HERE/'results_loso'
    classical=('lag4_matched_lr50','lag4_nuclear001_50','lag4_nuclear005_50','mssa8_mix25_lr50')
    for i in range(1,11):
        tag=f'loso_P{i:02}';dest=out/'predictions'/tag;dest.mkdir(parents=True,exist_ok=True)
        for name in classical:
            for suffix in ('','__balance'):
                source=ROOT/'mfm_models_20261110/results/predictions'/tag/(name+suffix+'.npz')
                shutil.copyfile(source,dest/source.name)
        with np.load(ROOT/'transfer_deep_dive_20261110/reference'/f'{tag}.npz') as z:
            m={k:z[k] for k in META}
            for key,name in [('old_champion','full_6272_recipe'),('champion','full_6323_recipe')]:
                np.savez_compressed(dest/(name+'.npz'),**m,proba=z[key])
        for old,new in [('early_and_filter_mean2__late_w0.3','full_6458_fixed30_recipe'),('early_and_filter_mean2__seed_agreement','full_6451_gate_recipe')]:
            source=HERE/'old_transfer_output/transfer-results/predictions'/tag/(old+'.npz')
            if source.exists():shutil.copyfile(source,dest/(new+'.npz'))
    summarize([('loso',out)],HERE/'reports_loso')
    print((HERE/'reports_loso/REPORT.md').read_text())

if __name__=='__main__':main()
