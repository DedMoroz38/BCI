from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from mixed_features import mix_covariances,fit_model,MatrixCache

def test_orthogonal_congruence_preserves_eigenvalues_and_physical_subsets():
 rng=np.random.default_rng(31);a=rng.normal(size=(6,6));b=rng.normal(size=(6,6))
 a=a@a.T+np.eye(6);b=b@b.T+np.eye(6)
 c=np.zeros((1,12,12));c[:,0::2,0::2]=a;c[:,1::2,1::2]=b
 q=np.kron(np.eye(6),np.array([[1.,1.],[1.,-1.]])/np.sqrt(2))
 r=mix_covariances(c)
 np.testing.assert_allclose(r,q@c@q.T,atol=1e-12)
 np.testing.assert_allclose(np.linalg.eigvalsh(r),np.linalg.eigvalsh(c),atol=1e-12)
 ix=np.array([0,1,6,7]);np.testing.assert_allclose(r[:,ix,:][:,:,ix],mix_covariances(c[:,ix,:][:,:,ix]),atol=1e-12)
 np.testing.assert_allclose(q@r@q.T,c,atol=1e-12)


def test_mixed_features_avoid_null_coordinate_amplification_on_fixture():
 sys.path.insert(0,str(Path(__file__).parents[1]))
 from test_temporal_features import windows
 from temporal_features import TemporalFeatures,compute_matrices
 tr=windows(12,32)
 c=mix_covariances(compute_matrices(tr.X,(4,40),'pyramid2_oas'))
 f=TemporalFeatures(representation='pyramid2_oas');f.fit_transform(tr,c)
 assert f.x_scaler.var_.min()>1e-12


def test_mixed_original_model_streaming_and_grouping():
 from test_temporal_features import windows
 from bci.evaluation import predict_streams
 from bci.models import forest
 tr,te=windows(12,33),windows(6,34)
 model=fit_model(tr,'forest12','pyramid2_mixed',n_trees=2,n_channels=3)
 original=forest.fit(tr,n_trees=2,n_channels=3)
 for a,b in zip(model.trees,original.trees):
  np.testing.assert_array_equal(a.physical_channels,b.channels)
  assert a.band==b.band and a.features.x_scaler.var_.min()>1e-12
 np.testing.assert_array_equal(predict_streams(model,te),model.tree_probas(te).mean(0).argmax(-1))
 np.testing.assert_array_equal(predict_streams(model,te),predict_streams(model,te))


def test_mixed_cached_live_models_agree(tmp_path):
 from test_temporal_features import windows
 from temporal_cache import initialize_cache,build_one,MatrixCache as BaseCache
 from bci.models.forest import BANDS
 from bci.evaluation import predict_streams
 from bci.data import concat
 tr,te=windows(12,35),windows(6,36)
 x=concat(tr,te).X
 initialize_cache(tmp_path,x,BANDS,{})
 for b in range(6):build_one(tmp_path,x,'pyramid2_oas',b,chunk_size=18)
 cache=MatrixCache.__new__(MatrixCache);cache.base=BaseCache(tmp_path)
 a=fit_model(tr,'forest12',cache=cache,n_trees=2,n_channels=3)
 b=fit_model(tr,'forest12',n_trees=2,n_channels=3)
 np.testing.assert_array_equal(predict_streams(a,te),predict_streams(b,te))
 for left,right in zip(a.trees,b.trees):np.testing.assert_allclose(left.clf.coef_,right.clf.coef_,atol=1e-10,rtol=1e-10)
