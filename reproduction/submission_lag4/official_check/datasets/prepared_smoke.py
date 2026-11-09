"""Local contract check with real-shape windows; not included in submission ZIP."""
from pathlib import Path
import numpy as np
from benchopt import BaseDataset
from benchmark_utils.data import make_loader,chs_info_from_names


class Dataset(BaseDataset):
    name='Prepared-PROTEUS-Smoke'
    requirements=[]

    def get_data(self):
        root=Path(__file__).resolve().parents[3]
        with np.load(root/'data/proteus_test.npz',allow_pickle=False) as f:
            data={k:f[k] for k in f.files}
        stream=np.char.add(np.char.add(data['subject'],'/'),data['session'])
        take=np.concatenate([np.flatnonzero(stream==s)[:12] for s in dict.fromkeys(stream)])
        mapping={};sid=np.array([mapping.setdefault(str(s),len(mapping)) for s in stream[take]])
        record_names=[f'{s}/{c}/{r}' for s,c,r in zip(stream[take],data['context'][take],data['run'][take])]
        mapping={};rid=np.array([mapping.setdefault(s,len(mapping)) for s in record_names])
        mapping={};ctx=np.array([mapping.setdefault(str(c),len(mapping)) for c in data['context'][take]])
        loader=make_loader(data['X'][take],data['y'][take],stream_id=sid,record_id=rid,context_id=ctx,onset=data['onset'][take],device='cpu')
        names=list(data['ch_names'])
        return dict(train_loader=None,test_loader=loader,n_classes=3,n_chans=41,n_times=480,sfreq=120.,ch_names=names,chs_info=chs_info_from_names(names),device='cpu')
