"""Fixed raw-forest substitutions before inherited causal TTA; no model fitting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPRO=HERE.parents[1]
CUSTOM=REPRO/'custom_validation_20261110'
sys.path.insert(0,str(CUSTOM))
import numpy as np
import pandas as pd
from core import META,metadata,subset,streams,balance,save,write_json
from scipy.special import softmax
from tta_core import spectral,features,causal_geometry
from evaluate import summarize

DATA=REPRO/'mfm_models_20261110/cache'
OLD=CUSTOM/'downloads/v1/custom-validation-results'
SYNERGY=HERE.parent/'cpu_synergy'
CACHE=HERE/'cache'
OUT=HERE/'results'
FORESTS={
    'full_nuclear001_recipe':('classical','lag4_nuclear001_50'),
    'full_nuclear005_recipe':('classical','lag4_nuclear005_50'),
    'full_mssa_nuclear001_recipe':('synergy','mssa8_mix25_nuclear001_50'),
    'full_mssa_nuclear005_recipe':('synergy','mssa8_mix25_nuclear005_50'),
}
TTA=dict(tau=.5,beta=1.,k=20,iters=5,shrink=.5,warm=20,ramp=0,strength=.5)

def validate_rows(actual,expected):
    for key in META:
        if not np.array_equal(actual[key],expected[key]):
            raise ValueError(f'Forecast metadata mismatch: {key}')

def source_statistics(logs,train_ids):
    return np.stack([np.mean(a[train_ids],axis=0,dtype='float64') for a in logs])

def fixed_geometry_predictions(logs,source,priors):
    """All variants share label-free geometry, each keeps its own raw prior."""
    f=features(logs,source,'lag_running',np.full((len(logs),3),1/3))
    return causal_geometry((f@f.T,),priors,[TTA])[0]

def validate_inputs():
    m=metadata(DATA);folds=json.loads((CUSTOM/'splits.json').read_text())['folds']
    # Verify window identity mapping, not historical model score parity.
    raw=np.load(DATA/'X.npy',mmap_mode='r');old_ids=np.load(DATA/'old_ids.npy')
    manifest=json.loads((REPRO/'temporal_spd_research/cache/manifest.json').read_text())
    for i,x in enumerate(raw):
        key=hashlib.blake2b(np.ascontiguousarray(x).tobytes(),digest_size=20).hexdigest()
        if key!=manifest['window_keys'][old_ids[i]]:raise ValueError(f'Cache row mapping mismatch {i}')
    required_bands=[(4,8),(8,13),(13,20),(20,30),(4,40)]
    bands=[tuple(b) for b in manifest['bands']]
    order=[bands.index(b) for b in required_bands]
    if order!=[1,2,3,4,0]:raise ValueError('Unexpected source covariance band mapping')
    for tag,record in folds.items():
        tr=np.asarray(record['train_ids']);te=np.asarray(record['test_ids']);me=subset(m,te)
        source_people=set(m['subject'][tr]);test_people=set(m['subject'][te])
        if len(source_people)!=8 or len(test_people)!=2 or source_people&test_people:
            raise ValueError(f'Invalid source exclusion {tag}')
        with np.load(CUSTOM/'classical_inputs'/f'{tag}.npz') as data:
            validate_rows(data,me)
            if not np.array_equal(data['data_ids'],te):raise ValueError(f'Classical IDs mismatch {tag}')
        with np.load(OLD/'predictions'/tag/'full_6458_fixed30_recipe.npz') as data:validate_rows(data,me)
        for family in ('swad_plain','point_meanstd','early_and_filter'):
            for seed in (0,1):
                folder=OLD/'models'/tag/family/f'seed{seed}'
                receipt=json.loads((folder/'complete.json').read_text())
                if set(receipt['train_people'])!=source_people or set(receipt['test_people'])!=test_people:
                    raise ValueError(f'Neural source exclusion mismatch {folder}')
                logits=np.load(folder/'logits.npy',mmap_mode='r')
                if logits.shape!=(len(te),3) or not np.isfinite(logits).all():raise ValueError(f'Invalid logits {folder}')
        for _,(kind,name) in FORESTS.items():
            if kind!='synergy':continue
            with np.load(SYNERGY/'results/predictions'/tag/(name+'.npz')) as data:
                validate_rows(data,me)
                if not np.array_equal(data['data_ids'],te):raise ValueError(f'Synergy IDs mismatch {tag}')
    receipt=dict(folds=len(folds),windows=len(raw),source_people=8,target_people=2,
                 metadata_aligned=True,window_identity_mapping=True,tta_bands=required_bands,
                 covariance_band_order=order,neural_logit_files=60,forest_variants=list(FORESTS))
    write_json(HERE/'INPUT_AUDIT.json',receipt)
    return m,folds,old_ids,order

def build_logs(old_ids,order):
    CACHE.mkdir(exist_ok=True);logs=[]
    for j,band in enumerate(order):
        dest=CACHE/f'laglog_{j}.npy';done=CACHE/f'laglog_{j}.json'
        if not done.exists():
            cov=np.load(REPRO/'temporal_spd_research/cache'/f'lag4_oas_band{band}.npy',mmap_mode='r')
            output=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(old_ids),82,82))
            for start in range(0,len(old_ids),32):
                ids=old_ids[start:start+32];output[start:start+len(ids)]=spectral(np.asarray(cov[ids],dtype='float64'),'log')
            output.flush();del output
            write_json(done,dict(band_index=band,windows=len(old_ids),dtype='float32',source='pure perwindow OAS log'))
            print('LOG CACHE',j,band,flush=True)
        logs.append(np.load(dest,mmap_mode='r'))
    return logs

def neural_probabilities(tag):
    values={}
    for family in ('swad_plain','point_meanstd','early_and_filter'):
        logits=[np.load(OLD/'models'/tag/family/f'seed{s}/logits.npy') for s in (0,1)]
        temperature=4 if family=='swad_plain' else 1
        values[family]=np.mean([softmax(z/temperature,axis=1) for z in logits],axis=0)
    return values

def run_fold(tag,record,m,logs):
    start=time.monotonic();tr=np.asarray(record['train_ids']);te=np.asarray(record['test_ids']);me=subset(m,te)
    existing=OUT/'predictions'/tag
    if all((existing/(name+'.npz')).exists() for name in list(FORESTS)+['full_6458_fixed30_recipe']):return
    nn=neural_probabilities(tag);forests=[]
    for kind,name in FORESTS.values():
        if kind=='classical':
            with np.load(CUSTOM/'classical_inputs'/f'{tag}.npz') as data:forests.append(data[name].copy())
        else:
            with np.load(SYNERGY/'results/predictions'/tag/(name+'.npz')) as data:forests.append(data['proba'].copy())
    prior=np.stack([.7*p+.3*nn['swad_plain'] for p in forests])
    sources=source_statistics(logs,tr)
    q0=np.empty_like(prior)
    for ids in streams(me):
        target=np.stack([a[te[ids]] for a in logs],axis=1).astype('float64')
        q0[:,ids]=fixed_geometry_predictions(target,sources,prior[:,ids])
    point=balance(nn['point_meanstd'],me);early=balance(nn['early_and_filter'],me)
    for i,name in enumerate(FORESTS):
        q1=.8*q0[i]+.2*point
        q2=.7*q1+.3*early
        save(OUT,tag,name,me,q2)
    with np.load(OLD/'predictions'/tag/'full_6458_fixed30_recipe.npz') as data:
        save(OUT,tag,'full_6458_fixed30_recipe',me,data['proba'].copy())
    write_json(HERE/'folds'/f'{tag}.json',dict(tag=tag,seconds=time.monotonic()-start,source_people=sorted(set(m['subject'][tr])),target_people=sorted(set(m['subject'][te])),variants=list(FORESTS)))
    print('FOLD COMPLETE',tag,round(time.monotonic()-start,1),flush=True)

def report():
    summarize([('pair8_2',OUT)],HERE/'validation')
    comparison=pd.read_csv(HERE/'validation/comparison.csv')
    people=pd.read_csv(HERE/'validation/per_person.csv')
    baseline='full_6458_fixed30_recipe'
    base=comparison.set_index('model').loc[baseline]
    comparison['delta_first_ba_pp']=100*(comparison.first_session_ba-base.first_session_ba)
    comparison['delta_first_nll']=comparison.first_nll-base.first_nll
    comparison.to_csv(HERE/'COMPARISON.csv',index=False)
    perbase=people[people.model==baseline].set_index('person')
    paired=[]
    for _,row in people.iterrows():
        paired.append(dict(model=row.model,person=row.person,first_ba=row.first_ba,
                           delta_first_ba_pp=100*(row.first_ba-perbase.loc[row.person,'first_ba']),
                           first_nll=row.first_nll,delta_first_nll=row.first_nll-perbase.loc[row.person,'first_nll']))
    pd.DataFrame(paired).to_csv(HERE/'PERSON_DELTAS.csv',index=False)
    text='# Fixed full-ensemble forest substitutions\n\n'
    text+='Same saved two-seed CNN forecasts, fixed inherited TTA, no fitting or weight tuning. Raw forest probabilities replaced before TTA. Primary: equal-person first-session BA, averaging two repeat metrics per person.\n\n'
    text+=comparison.to_string(index=False)+'\n\n'
    text+='Exploratory validation on previously studied public participants; no independent hidden-test claim. Source and target metadata audit: INPUT_AUDIT.json.\n'
    (HERE/'REPORT.md').write_text(text)
    print(comparison.to_string(index=False),flush=True)
    return comparison

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    begin=time.monotonic();m,folds,old_ids,order=validate_inputs()
    print('INPUTS VERIFIED',len(folds),'folds',flush=True)
    if args.validate_only:return
    status=json.loads((SYNERGY/'STATUS.json').read_text())
    if status['status']!='complete':raise RuntimeError('CPU synergy must finish before this process')
    logs=build_logs(old_ids,order)
    for i,(tag,record) in enumerate(folds.items()):
        run_fold(tag,record,m,logs)
        write_json(HERE/'STATUS.json',dict(completed_folds=i+1,planned_folds=len(folds),seconds=time.monotonic()-begin))
    report()
    write_json(HERE/'runtime.json',dict(status='complete',folds=10,variants=5,workers=1,blas_threads=1,seconds=time.monotonic()-begin))

if __name__=='__main__':main()
