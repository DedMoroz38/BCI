"""Pure single-window covariance cache. No labels, no source/equality gates."""
from pathlib import Path
import numpy as np
from common import filtering,oas,spectral,TTA_BANDS

def prep_job(args):
    folder,_,band=args;folder=Path(folder);X=np.load(folder/'X.npy',mmap_mode='r')
    dest=np.lib.format.open_memmap(folder/f'laglog_{band}.npy',mode='w+',dtype='float32',shape=(len(X),82,82))
    for i in range(0,len(X),16):
        x=filtering(X[i:i+16],TTA_BANDS[band]);x=np.stack([x[...,4:],x[...,:-4]],axis=-2).reshape(len(x),82,-1)
        dest[i:i+len(x)]=spectral(oas(x),'log')
    dest.flush();print('WINDOW CACHE',TTA_BANDS[band],'ready',flush=True)

