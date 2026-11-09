"""Prepare only the already verified public PROTEUS inputs for a private dataset."""
from pathlib import Path
import hashlib,json,shutil

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    out=HERE/'inputs';out.mkdir(exist_ok=True)
    sources=[ROOT/'data'/f'proteus_{s}.npz' for s in ['train','val','test']]
    cache=ROOT/'temporal_spd_research/cache'
    sources += [cache/'manifest.json']
    for b in range(6):sources += [cache/f'lag4_oas_band{b}{s}' for s in ['.npy','.complete.json']]
    hashes={}
    for source in sources:
        target=out/source.name;digest=sha(source)
        if not target.exists() or sha(target)!=digest:shutil.copyfile(source,target)
        hashes[target.name]=digest
    metadata=dict(id='den1shh/proteus-lag4-classifier-inputs',
                  title='PROTEUS Lag4 Classifier Experiment Inputs',
                  licenses=[dict(name='CC-BY-4.0')],
                  description='Private compute copy of public PROTEUS BCI Bordeaux CC-BY-4.0 data, DOI10.82901/nemar.nm000290. Includes the fixed original splits and deterministic OAS lag4 matrices; no new labels or private records.')
    (out/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest=dict(files_sha256=hashes,files=len(hashes),bytes=sum((out/n).stat().st_size for n in hashes),
                  source='PROTEUS BCI Bordeaux; DOI10.82901/nemar.nm000290',
                  data_policy='Existing public data only; private Kaggle compute copy; no credentials')
    (out/'input-integrity.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='files_sha256'},indent=2))

if __name__=='__main__':main()
