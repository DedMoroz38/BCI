"""Independently re-score stored predictions with the competition's metric code."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '2026-competition'))
from benchmark_utils.metrics import balanced_accuracy, group_scores

out = ROOT / 'results/eeg'
records = []
for p in sorted(out.glob('*_predictions.npz')):
    meta = json.loads(p.with_name(p.name.replace('_predictions.npz', '.json')).read_text())
    with np.load(p) as f:
        y, pred = f['y'], f['pred']
        stream = np.char.add(np.char.add(f['subject'], '/'), f['session'])
        cells = np.stack([stream, f['context']], axis=1)
        metrics = dict(session_mean=float(group_scores(balanced_accuracy, y, pred, stream).mean()),
                       cell_mean=float(group_scores(balanced_accuracy, y, pred, cells).mean()),
                       pooled=balanced_accuracy(y, pred), accuracy=float(np.mean(y == pred)))
        for key, value in metrics.items():
            if not np.isclose(value, meta[key], atol=1e-14, rtol=0):
                raise AssertionError(f'{p.name}: official {key} {value} != stored {meta[key]}')
        if meta['protocol'] in ['test', 'loso']:
            assert not set(meta['train_subjects']) & set(meta['eval_subjects'])
        assert len(y) == meta['eval_windows']
        assert set(np.unique(pred)) <= {0, 1, 2}
    records.append(meta)

if len(records) != 29:
    raise AssertionError(f'Expected 29 completed EEG jobs (24 LOSO + 3 later + 2 test), got {len(records)}')
for model in ['logreg', 'forest12', 'forest20']:
    assert sum(r['model'] == model and r['protocol'] == 'loso' for r in records) == 8
    assert sum(r['model'] == model and r['protocol'] == 'later' for r in records) == 1

result = {'verified_prediction_files': len(records),
          'official_metric_code': '2026-competition/benchmark_utils/metrics.py',
          'competition_commit': '2b53aec91c111f8b5423b6674f096c48cc1a99af',
          'metric_absolute_tolerance': 1e-14,
          'all_stored_scores_match_official_code': True,
          'split_leakage_check': 'passed for all LOSO and official test jobs'}
(out / 'verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
