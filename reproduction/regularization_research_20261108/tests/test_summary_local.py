"""End-to-end report regression using the actual previous GPU controls."""
import json,subprocess,sys,shutil
from pathlib import Path
import pandas as pd
import pytest
HERE=Path(__file__).resolve().parents[1]

def test_summary_reproduces_previous_two_seed_control_metric(tmp_path):
    wave=HERE.parent/'metric_research_20261108'
    root=wave/'downloads/gpu-v2/metric-results/models'
    if not root.exists():pytest.skip('Local-only previous-run artifact regression')
    for prediction in root.glob('*/shallow16_class/seed*/prediction.json'):
        # Prior predictions are only a numerical scorer fixture,not LS results.
        record=json.loads(prediction.read_text());record['config']='ls0.02'
        target=tmp_path/'models'/record['tag']/'ls0.02'/f'seed{record["seed"]}';target.mkdir(parents=True)
        for name in ('model.pt','alignment.npz','logits.npy','prediction.npz'):shutil.copyfile(prediction.parent/name,target/name)
        (target/'prediction.json').write_text(json.dumps(record))
    result=subprocess.run([sys.executable,'summarize.py','--root',str(tmp_path)],cwd=HERE,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    expected=pd.read_csv(wave/'to_human/GPU_ALL10_COMPARISON.csv').set_index('variant')
    row=pd.read_csv(tmp_path/'comparison.csv').query("seeds=='mean2'").iloc[0]
    assert row['mix_oof_cell']==pytest.approx(expected.loc['shallow16_class_mean2__mix_T4_balance1','official_oof_cell'],abs=1e-12)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['fits']==22 and summary['complete_mean2_configs']==1 and not summary['all_complete']
