import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pytest

import common


def test_tree_draws_match_saved_submission_for_actual_train_people():
    from bci.data import load_split
    train=load_split('train',common.ROOT/'data')
    specs=common.draw_specs(train)
    with np.load(common.ROOT/'submission_lag4/package/weights.npz') as w:
        np.testing.assert_array_equal([s['band'] for s in specs],w['tree_bands'])
        np.testing.assert_array_equal([s['matrix_channels'] for s in specs],w['channels'])
    assert all(len(s['channels'])==20 and len(s['matrix_channels'])==40 for s in specs)


def test_feature_cache_equals_live_streaming_with_resets():
    from bci.data import load_split
    from temporal_features import TemporalFeatures
    from temporal_cache import MatrixCache
    train=load_split('train',common.ROOT/'data')
    # Small real multi-session fixture with every class, avoiding a synthetic
    # covariance fixture that would miss ordering and stream-state errors.
    ids=[]
    for stream in list(train.streams())[:2]:
        for cls in range(3):ids.extend(stream[train.y[stream]==cls][:2])
    train=train.select(np.isin(np.arange(len(train)),ids))
    spec=common.draw_specs(train,n_trees=1)[0]
    matrices=MatrixCache(common.ROOT/'temporal_spd_research/cache')
    f,xt,xe=common.tree_features(train,train,spec,matrices)
    cov=matrices.lookup('lag4_oas',common.BANDS[spec['band']],train.X)
    cov=cov[:,spec['matrix_channels'],:][:,:,spec['matrix_channels']]
    live=np.zeros_like(xe)
    for ids in train.streams():
        session=f.new_session()
        for i in ids:live[i]=f.step(session,cov[i])[0]
    np.testing.assert_allclose(xe,live,rtol=0,atol=0)
    np.testing.assert_allclose(xt,xe,rtol=0,atol=0)
    assert xt.dtype==np.float64 and xt.shape==(len(train),820)

