import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Brain-Decoding/track2_bci'))
from bci.data import Windows
from bci.riemann import RiemannFeatures
from attention import AttentionSPD,encode_pack


def test_encoding_and_causal_prefix_session_reset():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(1,8,4,12)).astype('float32')
    model=AttentionSPD('attention_gram',[1],channels=4,samples=12)
    r=encode_pack(model,x)[0]
    assert np.isfinite(r).all() and np.linalg.eigvalsh(r).min()>0
    def windows(n):
        return Windows(X=x[0,:n],y=np.arange(n)%3,subject=np.repeat('p',n),
                       session=np.repeat('s',n),context=np.repeat('Graz',n),
                       run=np.ones(n),onset=np.arange(n),ch_names=np.arange(4))
    f=RiemannFeatures();f.fit_transform(windows(8),r)
    full=f.transform(windows(8),r)[0]
    prefix=f.transform(windows(4),r[:4])[0]
    np.testing.assert_allclose(prefix,full[:4],atol=1e-10)
    np.testing.assert_array_equal(full,f.transform(windows(8),r)[0])
    modified=x.copy();modified[:,4:]*=10
    np.testing.assert_allclose(encode_pack(model,modified)[0,:4],r[:4],atol=1e-12)


def test_trained_encoder_geometry_classifier_roundtrip_two_sessions(tmp_path):
    from attention import train_pack
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(10)
    x=rng.normal(size=(1,24,4,12)).astype('float32');y=np.tile(np.arange(3),8)
    fitted=train_pack(x,y,np.ones((1,24)),mode='attention_gram',tree_ids=[0],device='cpu',epochs=2,batch_size=8)
    model=fitted['model'];r=encode_pack(model,x)[0]
    w=Windows(X=x[0],y=y,subject=np.repeat('p',24),session=np.repeat(['a','b'],12),
              context=np.repeat('Graz',24),run=np.ones(24),onset=np.tile(np.arange(12),2),ch_names=np.arange(4))
    f=RiemannFeatures();features,_=f.fit_transform(w,r)
    clf=LogisticRegression(C=.01,max_iter=2000,class_weight='balanced').fit(features,y)
    p=clf.predict_proba(features)
    path=tmp_path/'encoder.pt';torch.save(model.state_dict(),path)
    restored=AttentionSPD('attention_gram',[0],channels=4,samples=12)
    restored.load_state_dict(torch.load(path,weights_only=True))
    r2=encode_pack(restored,x)[0]
    np.testing.assert_array_equal(r2,r)
    state=tmp_path/'geometry.npz'
    np.savez(state,prior=f.log_prior,mean=f.x_scaler.mean_,scale=f.x_scaler.scale_,coef=clf.coef_,intercept=clf.intercept_)
    with np.load(state) as saved:
        from bci.riemann import SessionRecentering
        z=np.empty_like(features)
        for idx in w.streams():
            session=SessionRecentering(saved['prior'],5)
            for i in idx:z[i]=(session(r2[i])[0]-saved['mean'])/saved['scale']
        logits=z@saved['coef'].T+saved['intercept'];logits-=logits.max(1,keepdims=True)
        pp=np.exp(logits);pp/=pp.sum(1,keepdims=True)
    np.testing.assert_allclose(p,pp,atol=1e-12)
    b=w.select(w.session=='b')
    np.testing.assert_allclose(f.transform(b,r[12:])[0],features[12:],atol=1e-12)
