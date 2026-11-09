import sys
import numpy as np
import common
from run import metrics


def test_report_metrics_equal_official_competition_grouping():
    sys.path.insert(0,str(common.ROOT/'2026-competition'))
    from benchmark_utils.metrics import balanced_accuracy,group_scores
    meta=dict(y=np.array([0,1,2,0,0,1,2,1]),
              subject=np.array(['a']*4+['b']*4),session=np.array(['1']*8),
              context=np.array(['Graz']*3+['BH']+['Graz']*4))
    pred=np.array([0,2,2,0,1,1,2,1]);p=np.eye(3)[pred]*.8+.2/3
    result=metrics(meta,p)
    stream=np.char.add(np.char.add(meta['subject'],'/'),meta['session'])
    cells=np.stack([stream,meta['context']],axis=1)
    assert result['session_mean']==float(group_scores(balanced_accuracy,meta['y'],pred,stream).mean())
    assert result['cell_mean']==float(group_scores(balanced_accuracy,meta['y'],pred,cells).mean())
    assert result['pooled']==balanced_accuracy(meta['y'],pred)
