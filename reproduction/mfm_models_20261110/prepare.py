"""Window-local caches only: no fitted source model or target-label use."""
from core import *
import hashlib
from numpy.lib.stride_tricks import sliding_window_view

def raw(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    if (cache/'raw_complete.json').exists():return
    parts=[]
    for split_name in ('train','val','test'):
        with np.load(ROOT/'data'/f'proteus_{split_name}.npz') as z:
            parts.append({k:z[k] for k in ('X',)+META})
    data={k:np.concatenate([a[k] for a in parts]) for k in ('X',)+META}
    np.save(cache/'X.npy',data['X']);np.savez(cache/'meta.npz',**{k:data[k] for k in META})
    old=json.loads((ROOT/'temporal_spd_research/cache/manifest.json').read_text())
    lookup={k:i for i,k in enumerate(old['window_keys'])}
    ids=np.array([lookup[hashlib.blake2b(np.ascontiguousarray(x).tobytes(),digest_size=20).hexdigest()] for x in data['X']])
    np.save(cache/'old_ids.npy',ids)
    out=np.lib.format.open_memmap(cache/'filtered.npy',mode='w+',dtype='float32',shape=data['X'].shape)
    for i in range(0,len(out),24):out[i:i+24]=filtering(data['X'][i:i+24],(4,40))
    out.flush();write_json(cache/'raw_complete.json',dict(windows=len(out),shape=list(out.shape)))
    print('CACHE raw and4–40Hz complete',flush=True)

def mssa_mixed(x,L=32):
    """Lecture14 horizontal mSSA, per-window temporal basis shared by channels."""
    h=sliding_window_view(x,L,axis=-1).swapaxes(-1,-2) # N,C,L,K
    mat=h.transpose(0,2,1,3).reshape(len(x),L,-1)
    gram=mat@mat.swapaxes(-1,-2)
    _,v=np.linalg.eigh((gram+gram.swapaxes(-1,-2))/2)
    K=h.shape[-1];counts=np.convolve(np.ones(L),np.ones(K))
    result=[]
    for rank,eta in ((8,.25),(16,.5)):
        U=v[:,:,-rank:];P=U@U.swapaxes(-1,-2)
        hh=(P@mat).reshape(len(x),L,x.shape[1],K).transpose(0,2,1,3)
        z=np.zeros_like(x)
        for j in range(L):z[...,j:j+K]+=hh[:,:,j,:]
        z/=counts
        result.append((1-eta)*x+eta*z)
    return result

def mssa_band(cache,band):
    cache=Path(cache);names=('mssa8_mix25','mssa16_mix50');done=cache/f'mssa_band{band}.json'
    if done.exists():return
    x=np.load(cache/'X.npy',mmap_mode='r');n=len(x)
    arrays=[np.lib.format.open_memmap(cache/f'{name}_band{band}.npy',mode='w+',dtype='float64',shape=(n,82,82)) for name in names]
    start=time.monotonic()
    for i in range(0,n,6):
        z=filtering(x[i:i+6],BANDS[band]);z-=z.mean(-1,keepdims=True)
        for dest,mixed in zip(arrays,mssa_mixed(z)):
            lag=np.stack([mixed[...,4:],mixed[...,:-4]],axis=-2).reshape(len(mixed),82,-1)
            dest[i:i+len(z)]=oas(lag)
    for a in arrays:a.flush()
    write_json(done,dict(seconds=time.monotonic()-start,band=band,variants=list(names)))
    print('CACHE mSSA band',band,round(time.monotonic()-start,1),'seconds',flush=True)
