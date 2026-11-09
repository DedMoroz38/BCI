"""Shared scheduler and exact outer split definitions; no CUDA at import."""
from pathlib import Path
import sys
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from configurations import CONFIGS
from common import sha

def fold_tags(w):return [f'loso_{s.split("/")[-1]}' for s in sorted(set(w.subject))]+['later_last']

def split_outer(w,tag):
    if tag.startswith('loso_'):
        person=tag[len('loso_'):];is_test=np.array([s.split('/')[-1]==person for s in w.subject])
        assert is_test.any()
    else:
        assert tag=='later_last';is_test=np.zeros(len(w),bool)
        for person in set(w.subject):
            sessions=sorted(set(w.session[w.subject==person]),key=lambda v:(len(v),v))
            if len(sessions)>1:is_test|=(w.subject==person)&(w.session==sessions[-1])
    return w.select(~is_test),w.select(is_test)

def schedule(tags):return [(c['name'],seed,tag) for c in CONFIGS for seed in (0,1) for tag in tags]

def validate_existing(path):
    record=json.loads(path.read_text())
    for name,key in [('prediction.npz','prediction_sha256'),('model.pt','model_sha256'),('alignment.npz','alignment_sha256'),('logits.npy','logits_sha256')]:
        assert sha(path.parent/name)==record[key],f'Corrupt completed artifact:{path.parent/name}'
    return record

def replay_diagnostics(expected,actual):
    assert expected.shape==actual.shape and np.isfinite(expected).all() and np.isfinite(actual).all()
    return dict(max_probability_delta=float(np.max(np.abs(actual-expected))),
        label_disagreements=int(np.sum(actual.argmax(1)!=expected.argmax(1))),windows=len(expected))
