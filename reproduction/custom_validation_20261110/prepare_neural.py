"""Archive exact context fits; add the old plain SWAD recipe without gates."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent

def extract(path,names):
    text=path.read_text();tree=ast.parse(text)
    return '\n\n'.join(ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)

def main():
    shared=extract(ROOT/'winner_research_20261109/gpu.py',{'atomic_torch','cell_weights','metrics','plot'})
    point=extract(ROOT/'winner_research_20261109/gpu.py',{'fit'}).replace('def fit(','def fit_point(',1)
    early=extract(ROOT/'transfer_deep_dive_20261110/run.py',{'fit'}).replace('def fit(','def fit_early(',1)
    classes=extract(ROOT/'winner_research_20261109/nets.py',{'Shallow','ContextShallow'})
    header='''"""Previous Shallow/meanstd/early recipes,refit only inside each source split."""
from pathlib import Path
from copy import deepcopy
import csv,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core import *
from averaging import StateAverage,LossValley,mean_states,clone_state
from transfer_nets import TransferNet
'''
    (HERE/'neural_training.py').write_text(header+classes+'\n\n'+shared+'\n\n'+point+'\n\n'+early+'\n\n'+(HERE/'neural_extra.txt').read_text())

if __name__=='__main__':main()
