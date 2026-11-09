from pathlib import Path
from collections import Counter
import hashlib,json,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('CLASSIFIER_OUTPUT',str(HERE/'results')))
CACHE=Path(os.environ.get('CLASSIFIER_CACHE',str(HERE/'cache')))
sys.path[:0]=[str(ROOT/'Brain-Decoding/track2_bci'),str(ROOT/'temporal_spd_research')]
from bci.models.forest import BANDS
from bci.data import concat,load_split
from bci.evaluation import leave_one_subject_out,later_sessions
from temporal_features import TemporalFeatures,expanded_channels

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)

def draw_specs(train,n_trees=50):
    rng=np.random.default_rng(0);subjects=sorted(set(train.subject));specs=[]
    for k in range(n_trees):
        band=int(rng.integers(len(BANDS)))
        channels=np.sort(rng.choice(train.X.shape[1],20,replace=False))
        counts=Counter(rng.choice(subjects,len(subjects)))
        specs.append(dict(band=band,channels=channels.tolist(),
                          matrix_channels=expanded_channels(channels).tolist(),
                          subject_counts={s:int(counts[s]) for s in subjects}))
    return specs

def tree_features(train,test,spec,cache,*,cov_train=None,cov_eval=None):
    band=BANDS[spec['band']];channels=np.asarray(spec['matrix_channels'])
    if cov_train is None:cov_train=cache.lookup('lag4_oas',band,train.X)
    if cov_eval is None:cov_eval=cache.lookup('lag4_oas',band,test.X)
    f=TemporalFeatures(band,5,representation='lag4_oas')
    xt,_=f.fit_transform(train,cov_train[:,channels,:][:,:,channels])
    xe,_=f.transform(test,cov_eval[:,channels,:][:,:,channels])
    if xt.shape!=(len(train),820) or xe.shape!=(len(test),820):raise RuntimeError('Feature dimensions changed')
    if not np.isfinite(xt).all() or not np.isfinite(xe).all():raise RuntimeError('Invalid features')
    return f,xt,xe

def folds(phase='all'):
    parts={s:load_split(s,ROOT/'data') for s in ['train','val','test']}
    dev=concat(parts['train'],parts['val'])
    jobs=[]
    if phase in ['all','cv']:
        jobs += [(f'loso_{s.split("/")[-1]}','loso',str(s),tr,te)
                 for s,tr,te in leave_one_subject_out(dev)]
        jobs += [('later_last','later','last',tr,te) for _,tr,te in later_sessions(dev)]
    if phase in ['all','test']:
        jobs += [('test_official','test','official',parts['train'],parts['test'])]
    return jobs

def control_path(protocol,fold):
    return ROOT/'temporal_spd_research/results/lag4_oas'/f'forest20_{protocol}_{fold.replace("/","_")}.json'

def metadata(windows):
    return {k:getattr(windows,k) for k in ['y','subject','session','context','run','onset']}
