"""Frozen simple-first grid; no hyperparameter decisions use outer labels."""
from copy import deepcopy
BASE=dict(n_filters=16,loss='class',ls=0.,augment={},average='none')
CONFIGS=[]

def add(name,stage=1,**changes):
    c=deepcopy(BASE);c.update(changes);c.update(name=name,stage=stage);CONFIGS.append(c)

for v in (.02,.05,.1,.2):add(f'ls{v}',ls=v)
for v in (.01,.03,.05):add(f'noise{v}',augment=dict(noise=v))
for key in ('global_gain','coordinate_gain'):
    for v in (.1,.2):add(f'{key}{v}',augment={key:v})
for v in (1,3,5):add(f'drop{v}',augment=dict(drop=v))
for v in (24,48):add(f'mask{v}',augment=dict(mask=v))
for v in (6,12):add(f'shift{v}',augment=dict(shift=v))
add('flip',augment=dict(flip=.5))
for v in (.2,.4):add(f'sameclass_mix{v}',augment=dict(mix=v))
for v in (.1,.3):add(f'mixstyle{v}',augment=dict(style=v))
for v in (3,5):add(f'window{v}',average='window',window=v)
for v in (.95,.99):add(f'ema{v}',average='ema',decay=v)
for v in (.05,.1):add(f'swad{v}',average='swad',ratio=v)
add('ls05_noise03',stage=2,ls=.05,augment=dict(noise=.03))
add('ls05_drop3',stage=2,ls=.05,augment=dict(drop=3))
add('ls05_gain20',stage=2,ls=.05,augment=dict(global_gain=.2))
add('noise03_drop3',stage=2,augment=dict(noise=.03,drop=3))
add('ls05_noise03_drop3',stage=2,ls=.05,augment=dict(noise=.03,drop=3))
add('ls05_window3',stage=2,ls=.05,average='window',window=3)
BY_NAME={c['name']:c for c in CONFIGS}
