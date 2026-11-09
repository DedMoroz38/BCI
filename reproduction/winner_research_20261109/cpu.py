"""Connectivity, FBCSP, random convolutions and frozen CNN geometry."""
import os
os.environ.update(NUMBA_NUM_THREADS='1',NUMBA_THREADING_LAYER='workqueue')
import argparse,json,time,traceback,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from common import *

FCBANDS=((8,13),(13,30),(8,30))

def prep_job(args):
    cache,kind,b=args;cache=Path(cache);X=np.load(cache/'X.npy',mmap_mode='r');n=len(X);d=X.shape[1]
    def array(name,shape):return np.lib.format.open_memmap(cache/(name+'.npy'),mode='w+',dtype='float32',shape=shape)
    if kind=='band':
        emp=array(f'emp_{b}',(n,d,d));shr=array(f'oas_{b}',(n,d,d))
        for i in range(0,n,24):
            z=filtering(X[i:i+24],BANDS[b]);z-=z.mean(-1,keepdims=True)
            emp[i:i+len(z)]=z@z.transpose(0,2,1)/z.shape[-1];shr[i:i+len(z)]=oas(z)
        emp.flush();shr.flush()
    elif kind=='phase':
        dest=array(f'plv_{b}',(n,d,d))
        for i in range(0,n,24):
            z=hilbert(filtering(X[i:i+24],FCBANDS[b]),axis=-1)
            u=z/np.maximum(np.abs(z),1e-12);dest[i:i+len(z)]=np.abs(u@u.conj().transpose(0,2,1)/u.shape[-1])
        dest.flush()
    elif kind=='coh':
        dest=[array(f'coh_{j}',(n,d,d)) for j in range(3)];win=np.hanning(120)
        for i in range(0,n,16):
            z=np.asarray(X[i:i+16],dtype='float64')
            segments=np.lib.stride_tricks.sliding_window_view(z,120,axis=-1)[:,:,::60,:].transpose(0,2,1,3)
            segments=(segments-segments.mean(-1,keepdims=True))*win
            F=np.fft.rfft(segments,axis=-1)[...,8:31]
            S=np.einsum('nwcf,nwdf->nfcd',F,F.conj())/F.shape[1]
            diagonal=np.real(np.diagonal(S,axis1=-2,axis2=-1))
            coh=np.abs(S)**2/np.maximum(diagonal[...,None]*diagonal[...,None,:],1e-30)
            freq=np.arange(8,31)
            for j,(lo,hi) in enumerate(FCBANDS):
                C=coh[:,(freq>=lo)&(freq<=hi)].mean(1).real
                dest[j][i:i+len(z)]=.9*C+.1*np.eye(d)
        for a in dest:a.flush()
    elif kind=='lag':
        dest=array(f'laglog_{b}',(n,2*d,2*d))
        for i in range(0,n,16):
            z=filtering(X[i:i+16],TTA_BANDS[b]);z=np.stack([z[...,4:],z[...,:-4]],axis=-2).reshape(len(z),2*d,-1)
            dest[i:i+len(z)]=spectral(oas(z),'log')
        dest.flush()
    print('CACHE',kind,b,'complete',flush=True)
    return kind,b

def configs():
    out=[]
    for b in range(3):out.append(dict(name=f'coh{b}_lr',kind='coh',band=b,head='lr'))
    out.append(dict(name='coh2_fgmdm',kind='coh',band=2,head='fgmdm'))
    for b in range(3):
        out.extend([dict(name=f'plv{b}_edges',kind='plv_edges',band=b,head='lr'),
                    dict(name=f'plv2_{b}_lr',kind='plv2',band=b,head='lr')])
    out.append(dict(name='plv2_2_fgmdm',kind='plv2',band=2,head='fgmdm'))
    out.extend(dict(name='power_'+s,kind='power',relative=(s=='relative'),head='lr') for s in ('absolute','relative'))
    out.extend(dict(name=f'csp_m{m}_ea{ea}',kind='csp',pairs=m,ea=ea,head='lr') for m in (1,2) for ea in (0,1))
    out.extend(dict(name=f'minirocket_{k}_ea{ea}',kind='rocket',n_features=k,ea=ea,head='lr') for k in (2000,10000) for ea in (0,1))
    return out

def ea_cov(C,meta,prior):
    out=np.empty_like(C)
    for ids in streams(meta):
        total=5*prior.copy();n=5
        for i in ids:
            total+=C[i];n+=1;w,v=np.linalg.eigh(total/n)
            W=(v/np.sqrt(np.maximum(w,w.max()*1e-10)))@v.T;out[i]=W@C[i]@W
    return out

def centered_cov(C,meta,source):
    L=spectral(C,'log');out=np.empty_like(C)
    for ids in streams(meta):
        totals=5*source+np.cumsum(L[ids],axis=0)
        W=spectral(-.5*totals/(5+np.arange(1,len(ids)+1))[:,None,None],'exp')
        out[ids]=W@C[ids]@W
    return out

def learn_csp(cache,meta,tr,te,cfg):
    aa=[];bb=[];saved={};d=41
    for b in range(5):
        C=np.load(Path(cache)/f'emp_{b}.npy',mmap_mode='r');a=np.asarray(C[tr],dtype='float64');v=np.asarray(C[te],dtype='float64')
        if cfg['ea']:
            prior=a.mean(0);a=ea_cov(a,subset(meta,tr),prior);v=ea_cov(v,subset(meta,te),prior)
        classes=[]
        for c in range(3):
            means=[]
            for person in np.unique(meta['subject'][tr]):
                ids=(meta['subject'][tr]==person)&(meta['y'][tr]==c)
                if ids.any():means.append(a[ids].mean(0))
            R=np.mean(means,axis=0);classes.append(.9*R+.1*np.trace(R)/d*np.eye(d))
        filters=[]
        for c in range(3):
            pos=classes[c];neg=np.mean([r for j,r in enumerate(classes) if j!=c],axis=0)
            _,W=eigh(pos,pos+neg);m=cfg['pairs'];filters.append(W[:,np.r_[np.arange(m),np.arange(d-m,d)]].T)
        W=np.concatenate(filters);saved[f'band{b}']=W
        def project(C):
            p=np.maximum(np.einsum('fc,ncd,fd->nf',W,C,W),1e-20)
            return np.log(p/p.sum(1,keepdims=True))
        aa.append(project(a));bb.append(project(v))
    return np.concatenate(aa,axis=1),np.concatenate(bb,axis=1),saved

def expert(cache,meta,tr,te,cfg):
    cache=Path(cache);kind=cfg['kind'];y=meta['y'][tr];mt= subset(meta,tr);me=subset(meta,te);saved={}
    if kind in ('coh','plv2','plv_edges'):
        stem='coh' if kind=='coh' else 'plv';C=np.load(cache/f"{stem}_{cfg['band']}.npy",mmap_mode='r')
        a=np.asarray(C[tr],dtype='float64');b=np.asarray(C[te],dtype='float64')
        if kind=='plv_edges':
            ij=np.triu_indices(41,1);a=a[:,ij[0],ij[1]];b=b[:,ij[0],ij[1]]
        else:
            if kind=='plv2':a=.9*a*a+.1*np.eye(41);b=.9*b*b+.1*np.eye(41)
            source=spectral(a,'log').mean(0);saved['log_prior']=source
            if cfg['head']=='fgmdm':a=centered_cov(a,mt,source);b=centered_cov(b,me,source)
            else:a=tangent(a,mt,source);b=tangent(b,me,source)
    elif kind=='power':
        z=[]
        for band in range(5):
            C=np.load(cache/f'emp_{band}.npy',mmap_mode='r');z.append(np.maximum(np.diagonal(C,axis1=-2,axis2=-1),1e-20))
        z=np.stack(z,axis=1)
        if cfg['relative']:z=z/z.sum(1,keepdims=True)
        z=np.log(z).reshape(len(z),-1);a=z[tr];b=z[te]
    elif kind=='csp':a,b,saved=learn_csp(cache,meta,tr,te,cfg)
    else:
        sys.path.insert(0,str(HERE/'vendor/minirocket/code'))
        import minirocket_multivariate as rocket
        from numba import njit
        @njit
        def seed_numba():np.random.seed(0)
        seed_numba()
        X=np.load(cache/'X.npy',mmap_mode='r');x=filtering(X[tr],(4,40));v=filtering(X[te],(4,40))
        if cfg['ea']:
            prior=np.mean(x@x.transpose(0,2,1)/x.shape[-1],axis=0);x=align(x,mt,prior);v=align(v,me,prior)
        x=np.ascontiguousarray(x,dtype='float32');v=np.ascontiguousarray(v,dtype='float32')
        params=rocket.fit(x,num_features=cfg['n_features']);a=rocket.transform(x,params);b=rocket.transform(v,params)
        saved={f'parameter{i}':p for i,p in enumerate(params)}
    if cfg['head']=='fgmdm':
        from pyriemann.classification import FgMDM
        model=FgMDM(metric='logeuclid');model.fit(a,y)
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=600,solver='lbfgs'))
        model.fit(a,y)
    return model.predict_proba(b),model,saved

def frozen_geometry(out,cache,meta,tr,te,tag,deadline):
    import torch
    from nets import Shallow
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    torch.set_num_threads(1);ref=reference(tag);a=[];b=[]
    X=np.load(Path(cache)/'X.npy',mmap_mode='r')
    for seed in (0,1):
        folder=HERE/'frozen'/tag/f'seed{seed}'
        model=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        net=Shallow(**model['config']);net.load_state_dict(model['state']);net.eval()
        with np.load(folder/'alignment.npz') as z:prior=z['cov_prior']
        def transform(ids):
            xf=filtering(X[ids],(1,58));xf=align(xf,subset(meta,ids),prior);h=[]
            with torch.no_grad():
                for i in range(0,len(xf),64):h.append(net.encode(torch.from_numpy(xf[i:i+64])).numpy())
            return np.concatenate(h)
        a.append(transform(tr));b.append(transform(te))
    a=np.mean(a,axis=0);b=np.mean(b,axis=0);scaler=StandardScaler().fit(a);a=scaler.transform(a);b=scaler.transform(b)
    pca=PCA(n_components=16,svd_solver='randomized',random_state=0).fit(a)
    lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto',n_components=2).fit(a,meta['y'][tr])
    embeddings={'pca16':pca.transform(b),'lda2':lda.transform(b)}
    import joblib
    folder=Path(out)/'geometry'/tag;folder.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(scaler=scaler,pca=pca,lda=lda),folder/'projections.joblib')
    grams=tta_grams(cache,meta,tr,te);mt=subset(meta,te)
    for kind,w in (('pca16',.25),('pca16',.5),('lda2',.25)):
        result=np.empty((len(te),3));z=embeddings[kind]
        for idx,(g,) in grams:
            q=z[idx]@z[idx].T
            # causal_geometry averages normalized views. Scaling one view by
            # its requested weight is handled explicitly for every prefix.
            final=ref['fallback'][idx].copy()
            from scipy.linalg import eigh as ev
            from tta_core import solve_many
            for n in range(20,len(idx)+1):
                g0=g[:n,:n];g1=q[:n,:n];G=(1-w)*g0/max(np.trace(g0)/n,1e-12)+w*g1/max(np.trace(g1)/n,1e-12)
                G=G-G.mean(0)[None]-G.mean(1)[:,None]+G.mean();k=min(20,n//3)
                _,V=ev(G,subset_by_index=[n-k,n-1],check_finite=False);P=V[:,::-1]*np.sqrt(n)
                case=dict(tau=.5,beta=1.,shrink=.5)
                new=solve_many(P,np.log(np.maximum(ref['mixed_raw'][idx[:n]][None],1e-9)),[case])[0]
                final[n-1]=.5*new+.5*ref['fallback'][idx[n-1]]
            result[idx]=final
        save_prediction(out,tag,f'geometry_{kind}_w{w}',mt,result)
    np.savez_compressed(folder/'hidden.npz',source=a,target=b,train_ids=tr,test_ids=te)

def fold_job(args):
    tag,cache,out,deadline=args;cache=Path(cache);out=Path(out)
    with np.load(cache/'meta.npz') as z:meta={k:z[k] for k in META}
    tr,te=train_test(meta,tag);mt=subset(meta,te);ref=reference(tag);raw=[];names=[];failures=[]
    for w in (.25,.5,.75):save_prediction(out,tag,f'existing_B_S_w{w}',mt,(1-w)*ref['champion']+w*ref['stable'])
    for c in range(3):
        alpha=np.full(3,.5);alpha[c]=.25;q=(1-alpha)*ref['fallback']+alpha*ref['full_trans'];q/=q.sum(1,keepdims=True)
        save_prediction(out,tag,f'class_strength_{c}',mt,q)
    for cfg in configs():
        if time.monotonic()>deadline:break
        try:
            begin=time.monotonic();p,model,params=expert(cache,meta,tr,te,cfg)
            save_expert(out,tag,cfg['name'],mt,p,config=cfg,fit_predict_seconds=time.monotonic()-begin)
            import joblib
            folder=out/'models'/tag;folder.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,folder/(cfg['name']+'.joblib'));np.savez_compressed(folder/(cfg['name']+'_params.npz'),**params)
            for w in (.1,.2):raw.append((1-w)*ref['mixed_raw']+w*p);names.append(cfg['name']+f'__prior_w{w}')
        except Exception as e:
            failures.append(dict(config=cfg['name'],error=str(e),traceback=traceback.format_exc()));print('EXPERT_FAILED',tag,cfg['name'],str(e),flush=True)
    if raw and time.monotonic()<deadline:
        try:
            grams=tta_grams(cache,meta,tr,te);q=apply_tta_many(grams,mt,np.stack(raw),deadline)
            for name,p in zip(names,q):save_prediction(out,tag,name,mt,p,fusion='prior_then_transduction')
        except Exception as e:failures.append(dict(config='transduction',error=str(e),traceback=traceback.format_exc()))
    if time.monotonic()<deadline:
        try:frozen_geometry(out,cache,meta,tr,te,tag,deadline)
        except Exception as e:failures.append(dict(config='frozen_geometry',error=str(e),traceback=traceback.format_exc()));print('GEOMETRY_FAILED',tag,str(e),flush=True)
    write_json(out/'fold_status'/f'{tag}.json',dict(failures=failures,experts=len(names)//2,budget_exhausted=time.monotonic()>deadline))
    print('FOLD FINISHED',tag,'expert failures',len(failures),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',default='/kaggle/input');p.add_argument('--out',default='/kaggle/working/winner-cpu-results')
    p.add_argument('--cache',default='/tmp/winner-cache');p.add_argument('--minutes',type=float,default=240);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True);Path(a.out).mkdir(parents=True,exist_ok=True);start=time.monotonic();deadline=start+a.minutes*60
    data=load_data(a.data_root);np.save(cache/'X.npy',data.pop('X'));np.savez(cache/'meta.npz',**data);del data
    tasks=[(a.cache,k,b) for k,bs in [('band',range(5)),('phase',range(3)),('coh',range(1)),('lag',range(5))] for b in bs]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(prep_job,tasks))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(fold_job,(tag,a.cache,a.out,deadline)):tag for tag in TAGS}
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:write_json(Path(a.out)/'fold_status'/f'{futures[f]}.json',dict(error=str(e),traceback=traceback.format_exc()))
    summarize(a.out);write_json(Path(a.out)/'runtime.json',dict(seconds=time.monotonic()-start,planned_classical_configs=len(configs())))

if __name__=='__main__':main()
