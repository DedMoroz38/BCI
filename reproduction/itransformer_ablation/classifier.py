"""Minimal classification adapter of pinned official iTransformer Model."""
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor/itransformer'))
from model.iTransformer import Model as AuthorModel


def config():
    return SimpleNamespace(seq_len=480,pred_len=1,output_attention=False,use_norm=False,
              d_model=64,n_heads=4,d_ff=128,e_layers=1,dropout=.2,activation='gelu',
              factor=5,embed='fixed',freq='h',class_strategy='projection')

class EEGiTransformer(AuthorModel):
    def __init__(self):
        super().__init__(config())
        # Replace only forecasting output; retain exact author embedding/encoder.
        self.projector=nn.Linear(41*64,3)
        self.register_buffer('input_scale',torch.ones(()))
    def hidden(self,x):
        if x.ndim!=3 or tuple(x.shape[1:])!=(41,480):raise ValueError('Expected Bx41x480 without lag copies')
        x=(x-x.mean(-1,keepdim=True))/self.input_scale
        h,_=self.encoder(self.enc_embedding(x.transpose(1,2),None),attn_mask=None)
        return h
    def forward(self,x):return self.projector(self.hidden(x).flatten(1))

@torch.no_grad()
def fit_scale(model,training_x):
    a=training_x-training_x.mean(-1,keepdim=True)
    model.input_scale.copy_(a.square().mean().sqrt().clamp_min(1e-6))
