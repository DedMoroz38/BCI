"""Approved participant-first validation, with source-only fitting."""
from base import *
TTA_BANDS=((4,8),(8,13),(13,20),(20,30),(4,40))

def split(meta,tag):
    record=json.loads((HERE/'splits.json').read_text())['folds'][tag]
    return np.asarray(record['train_ids']),np.asarray(record['test_ids'])

def pair_tags():return list(json.loads((HERE/'splits.json').read_text())['folds'])

def get_aligned(X,m,tr,te,band=(1,58)):
    a=filtering(X[tr],band);b=filtering(X[te],band)
    prior=np.mean(a@a.transpose(0,2,1)/a.shape[-1],axis=0)
    return ea(a,subset(m,tr),prior),ea(b,subset(m,te),prior),prior

def class_weights(y):
    counts=np.bincount(y,minlength=3);return len(y)/(3*np.maximum(counts[y],1))
