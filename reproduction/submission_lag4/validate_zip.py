"""Load only the ZIP in a read-only temp folder; replay the official stream."""
from pathlib import Path
import sys,importlib.util,importlib.abc,json,hashlib,zipfile,tempfile,os,time
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'2026-competition'))
from benchmark_utils.streaming import predict_streams
from benchmark_utils.data import make_loader
from benchmark_utils.metrics import balanced_accuracy,group_scores


class NoResearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'bci','pyriemann','temporal_features','temporal_cache','references','mixed_features'}:
            raise ImportError('Research dependency forbidden in packaged inference: '+fullname)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ids(values):
    mapping={};return np.array([mapping.setdefault(str(v),len(mapping)) for v in values],dtype=np.int64)


def main():
    archive=HERE/'riemann_forest_lag4.zip'
    expected_files={'submission.py','weights.npz','model_info.json'}
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==expected_files
        assert z.testzip() is None
        with tempfile.TemporaryDirectory(prefix='lag4-submission-') as directory:
            unpacked=Path(directory);z.extractall(unpacked)
            info=json.loads((unpacked/'model_info.json').read_text())
            assert sha(unpacked/'weights.npz')==info['weights_sha256']
            before={name:sha(unpacked/name) for name in expected_files}
            with np.load(ROOT/'data/proteus_test.npz',allow_pickle=False) as data:
                arrays={k:data[k] for k in data.files}
            meta=dict(submission_dir=unpacked,n_chans=41,n_times=480,n_classes=3,sfreq=120.,ch_names=list(arrays['ch_names']),device='cpu')
            for name in expected_files:(unpacked/name).chmod(0o444)
            unpacked.chmod(0o555)
            try:
                sys.meta_path.insert(0,NoResearchImports())
                spec=importlib.util.spec_from_file_location('submission',unpacked/'submission.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                model=module.Solver.get_instance().load_model(meta)
                streams=np.char.add(np.char.add(arrays['subject'],'/'),arrays['session'])
                records=np.array([f'{s}/{c}/{r}' for s,c,r in zip(streams,arrays['context'],arrays['run'])])
                loader=make_loader(arrays['X'],arrays['y'],record_id=ids(records),stream_id=ids(streams),context_id=ids(arrays['context']),onset=arrays['onset'],device='cpu')
                start=time.monotonic()
                with threadpool_limits(limits=1):y,pred,stream_info=predict_streams(model,loader)
                seconds=time.monotonic()-start;pred=pred.reshape(-1)
                reference_dir=ROOT/'temporal_spd_research/results/lag4_oas'
                with np.load(reference_dir/'forest20_test_official_predictions.npz') as expected:
                    np.testing.assert_array_equal(y,expected['y'])
                    np.testing.assert_array_equal(pred,expected['pred'])
                    for key in ['subject','session','context','run','onset']:np.testing.assert_array_equal(arrays[key],expected[key])
                cells=np.stack([stream_info['stream_id'],stream_info['context_id']],axis=1)
                metrics=dict(cell_mean=float(group_scores(balanced_accuracy,y,pred,cells).mean()),session_mean=float(group_scores(balanced_accuracy,y,pred,stream_info['stream_id']).mean()),pooled=balanced_accuracy(y,pred),accuracy=float(np.mean(y==pred)))
                expected_meta=json.loads((reference_dir/'forest20_test_official.json').read_text())
                for k,v in metrics.items():assert np.isclose(v,expected_meta[k],atol=1e-14,rtol=0),(k,v)
                new_window=arrays['X'][:1].copy();new_window[0,0,0]+=.123
                assert model.predict(new_window).shape==(1,)
                assert {name:sha(unpacked/name) for name in expected_files}==before
            finally:
                unpacked.chmod(0o755)
                for name in expected_files:(unpacked/name).chmod(0o644)
    report=dict(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=sorted(expected_files),all755_predictions_match_reference=True,windows=len(pred),streams=len(np.unique(streams)),official_streaming_used=True,read_only_package_unchanged=True,research_imports_blocked=True,unknown_window_supported=True,metrics=metrics,inference_seconds=seconds,weights_sha256=info['weights_sha256'])
    (HERE/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
