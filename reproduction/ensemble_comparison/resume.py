"""Import verified v2 checkpoints with explicit lineage,then run missing jobs."""
import json,os,shutil
from pathlib import Path
import numpy as np
import run

def main():
    candidates=[Path(os.environ['ENSEMBLE_PARENT'])/'manifest.json'] if os.environ.get('ENSEMBLE_PARENT') else list(Path('/kaggle/input').rglob('ensemble-results/manifest.json'))
    if len(candidates)!=1:raise RuntimeError(f'Need one v2 output source,found{len(candidates)}')
    parent=candidates[0].parent
    old=json.loads((parent/'manifest.json').read_text());oldhash=run.sha(parent/'manifest.json')
    if oldhash!=os.environ['ENSEMBLE_PARENT_MANIFEST_SHA256']:raise RuntimeError('Wrong parent version')
    h=run.initialize();current=json.loads((run.ARTIFACT/'manifest.json').read_text())
    for key in ('data','cache','references','versions','variants'):
        if current[key]!=old[key]:raise RuntimeError('Resume changed numerical inputs: '+key)
    for key in ('n_trees','n_channels','n_subject_trees','C','prior_weight','seed','deep_seeds','deep_epochs','torch_threads','blas_threads','deep_device'):
        if current['config'][key]!=old['config'][key]:raise RuntimeError('Resume changed hyperparameter: '+key)
    for path,digest in old['code'].items():
        if '/bci/' in path or '/temporal_spd_research/' in path:
            if current['code'].get(path)!=digest:raise RuntimeError('Upstream numerical source changed: '+path)
    imported={};counts={};missing_warnings=[]
    for folder in sorted((parent/'results').iterdir()):
        if not folder.is_dir():continue
        matching=[f for f in run.folds('cv') if f[0]==folder.name]
        if len(matching)!=1:raise RuntimeError('Unexpected parent fold')
        _,_,test=matching[0]
        receipt=json.loads((folder/'parts.sha.json').read_text())
        if receipt!={'sha256':run.sha(folder/'parts.npz'),'manifest':oldhash}:raise RuntimeError('Changed checkpoint')
        records={}
        for path in folder.glob('*.json'):
            if path.stem!='baseline' and not path.stem.startswith('variant'):continue
            r=json.loads(path.read_text());pred=path.with_suffix('.npz')
            if r['manifest_sha256']!=oldhash or r['prediction_sha256']!=run.sha(pred):raise RuntimeError('Changed parent result')
            with np.load(pred) as f:
                run.check_metadata(test,f)
                for key,value in run.measures(test,f['proba']).items():np.testing.assert_allclose(value,r[key],rtol=0,atol=1e-12)
            records[path.name]=r
        dest=run.OUT/folder.name
        if dest.exists():raise RuntimeError('Recovery output not empty')
        shutil.copytree(folder,dest)
        for path in folder.iterdir():
            if path.is_file():imported[str(path.relative_to(parent))]=run.sha(path)
        for name,r in records.items():
            r['parent_record_sha256']=run.sha(folder/name);r['parent_manifest_sha256']=oldhash;r['manifest_sha256']=h
            run.write_json(dest/name,r)
        run.write_json(dest/'parts.sha.json',dict(sha256=receipt['sha256'],manifest=h))
        counts[folder.name]=len(records)
        if not (folder/'warnings.json').exists():missing_warnings.append(folder.name)
    shutil.copy2(parent/'manifest.json',run.ARTIFACT/'parent-manifest.json')
    run.write_json(run.ARTIFACT/'resume-lineage.json',dict(parent_manifest=oldhash,recovery_manifest=h,files=imported,records=counts,original_training_warnings_unavailable=missing_warnings))
    print('IMPORTED_CHECKPOINTS',len(counts),'RESULTS',sum(counts.values()),'NO RETRAINING OF FINISHED FITS',flush=True)
    run.main()

if __name__=='__main__':main()
