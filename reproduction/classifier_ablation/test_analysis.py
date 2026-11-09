import numpy as np
from sklearn.metrics import log_loss
import analyze
from pathlib import Path
import subprocess,sys


def test_standalone_analysis_loads_its_own_runner_despite_upstream_name_collision():
    source="import analyze;from pathlib import Path;assert Path(analyze.config_hash.__code__.co_filename).resolve()==Path('run.py').resolve()"
    subprocess.run([sys.executable,'-c',source],cwd=Path(__file__).resolve().parent,check=True)


def test_probability_scores_are_independently_recomputed():
    y=np.array([0,2,1]);p=np.array([[.7,.2,.1],[.1,.2,.7],[.3,.4,.3]])
    got=analyze.probability_scores(y,p)
    np.testing.assert_allclose(got['log_loss'],log_loss(y,p,labels=[0,1,2]),atol=1e-14)
    expected=sum(sum((row[j]-(label==j))**2 for j in range(3)) for row,label in zip(p,y))/len(y)
    np.testing.assert_allclose(got['brier'],expected,atol=1e-14)


def test_selection_stability_uses_paired_people_and_alphabetical_ties():
    scores={'z':[.6,.6,.6],'a':[.6,.6,.6],'loser':[.4,.4,.4]}
    samples=np.array([[0,0,1],[2,1,0],[2,2,2]])
    got=analyze.selection_stability(scores,['p1','p2','p3'],samples)
    assert got['leave_one_subject_out_wins']=={'a':3,'loser':0,'z':0}
    assert got['paired_bootstrap_winner_fraction']=={'a':1.,'loser':0.,'z':0.}
    assert [r['omitted_subject'] for r in got['leave_one_subject_out']]==['p1','p2','p3']
