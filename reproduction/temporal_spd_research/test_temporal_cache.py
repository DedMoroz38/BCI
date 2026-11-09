"""Exact per-window caching must preserve matrices and model predictions."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
try:
    from temporal_cache import initialize_cache,build_one,MatrixCache
except ImportError:
    MatrixCache=None

@pytest.mark.parametrize('representation',['lag4_oas','stationary_oas','pyramid2_oas','quadrature_oas','path_integral_oas','signature2_oas'])
def test_cached_lookup_matches_live_order_and_rejects_unknown_window(tmp_path,representation):
    assert MatrixCache is not None, 'Exact matrix cache implementation is missing'
    from temporal_features import compute_matrices
    X=np.random.default_rng(11).normal(size=(3,4,480)).astype('float32')
    initialize_cache(tmp_path,X,[(4,40)],{'fixture':'test'})
    build_one(tmp_path,X,representation,0,chunk_size=2)
    cache=MatrixCache(tmp_path)
    shuffled=X[[2,0,2]]
    np.testing.assert_allclose(cache.lookup(representation,(4,40),shuffled),compute_matrices(shuffled,(4,40),representation),atol=1e-12,rtol=1e-12)
    assert copy.deepcopy(cache) is cache
    unknown=X[:1].copy();unknown[0,0,0]+=1
    with pytest.raises(KeyError):cache.lookup(representation,(4,40),unknown)
    with pytest.raises(TypeError):cache.lookup(representation,(4,40),X.astype('float64'))


def test_cache_resume_preserves_finished_matrix_and_detects_input_change(tmp_path):
    assert MatrixCache is not None, 'Exact matrix cache implementation is missing'
    X=np.random.default_rng(12).normal(size=(3,4,480)).astype('float32')
    initialize_cache(tmp_path,X,[(4,40)],{})
    a=build_one(tmp_path,X,'path_integral_oas',0,chunk_size=2)
    b=build_one(tmp_path,X,'path_integral_oas',0,chunk_size=1)
    assert a['sha256']==b['sha256']
    changed=X.copy();changed[0,0,0]+=1
    with pytest.raises(RuntimeError):initialize_cache(tmp_path,changed,[(4,40)],{})

@pytest.mark.parametrize('representation',['lag4_oas','stationary_oas','pyramid2_oas','quadrature_oas','path_integral_oas','signature2_oas'])
def test_full_model_cached_and_live_training_and_streaming_match(tmp_path,representation):
    assert MatrixCache is not None, 'Exact matrix cache implementation is missing'
    from bci.data import Windows
    from bci.models.forest import BANDS
    from bci.evaluation import predict_streams
    from temporal_features import fit_model
    rng=np.random.default_rng(13)
    X=rng.normal(size=(18,6,480)).astype('float32')
    def part(rows):
        n=len(rows)
        return Windows(X=X[rows],y=np.arange(n)%3,subject=np.repeat(['a','b'],n//2),
            session=np.full(n,'1'),context=np.full(n,'Graz'),run=np.ones(n,int),
            onset=np.arange(n,dtype=float),ch_names=np.array([str(i) for i in range(6)]))
    tr,te=part(np.arange(12)),part(np.arange(12,18))
    initialize_cache(tmp_path,X,BANDS,{})
    for b in range(6):build_one(tmp_path,X,representation,b,chunk_size=6)
    cached=fit_model(tr,'forest12',representation,cache=MatrixCache(tmp_path),n_trees=2,n_channels=3)
    live=fit_model(tr,'forest12',representation,n_trees=2,n_channels=3)
    np.testing.assert_array_equal(predict_streams(cached,te),predict_streams(live,te))
    for a,b in zip(cached.trees,live.trees):
        np.testing.assert_allclose(a.clf.coef_,b.clf.coef_,atol=1e-10,rtol=1e-10)


def test_cache_recovers_if_process_stops_after_final_file_before_receipt(tmp_path):
    assert MatrixCache is not None, 'Exact matrix cache implementation is missing'
    X=np.random.default_rng(14).normal(size=(2,4,480)).astype('float32')
    initialize_cache(tmp_path,X,[(4,40)],{})
    before=build_one(tmp_path,X,'path_integral_oas',0,chunk_size=1)
    (tmp_path/'path_integral_oas_band0.complete.json').unlink()
    after=build_one(tmp_path,X,'path_integral_oas',0,chunk_size=1)
    assert before['sha256']==after['sha256']
