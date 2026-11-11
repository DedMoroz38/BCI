"""Averaging policies from our prior SWAD experiment, unchanged."""
from collections import deque
import torch
import numpy as np

def clone_state(state):return {k:v.detach().cpu().clone() for k,v in state.items()}

def mean_states(states):
    assert states
    result=clone_state(states[-1])
    for key,value in result.items():
        if value.is_floating_point():
            value.copy_(states[0][key]);
            for n,state in enumerate(states[1:],2):value.add_((state[key]-value)/n)
    return result

class StateAverage:
    def __init__(self,kind='mean',decay=0.):self.kind=kind;self.decay=decay;self.n=0;self.state=None
    def update(self,state):
        if self.state is None:self.state=clone_state(state)
        else:
            factor=1-self.decay if self.kind=='ema' else 1/(self.n+1)
            for k,v in self.state.items():
                if v.is_floating_point():v.add_((state[k].detach().cpu()-v)*factor)
                else:v.copy_(state[k].detach().cpu())
        self.n+=1
    def result(self):return clone_state(self.state)
    def state_dict(self):return dict(kind=self.kind,decay=self.decay,n=self.n,state=self.state)
    def load_state_dict(self,s):self.kind=s['kind'];self.decay=s['decay'];self.n=s['n'];self.state=s['state']

class LossValley:
    """Scalar/epoch-index adaptation of khanrc/swad LossValley (MIT).

    Dense parameter segment means are maintained by the training loop.
    The original source/license are archived under literature/vendor/.
    """
    def __init__(self,n_converge=3,n_tolerance=6,ratio=.05):
        self.nc=n_converge;self.nt=n_tolerance;self.ratio=ratio
        self.cq=deque(maxlen=n_converge);self.sq=deque(maxlen=n_tolerance)
        self.start=None;self.threshold=None;self.dead=False;self.selected=[]
    def update(self,epoch,loss):
        if self.dead:return
        item=(epoch,float(loss));self.cq.append(item);self.sq.append(item)
        if self.start is None:
            if len(self.cq)<self.nc:return
            if int(np.argmin([v for _,v in self.cq]))==0:
                # Author AveragedModel starts with n_averaged=0: the first
                # update replaces its initial copied segment. Preserve that
                # behavior,including repeated segment weights in later queues.
                self.start=self.cq[0][0];self.selected=[]
                self.threshold=np.mean([v for _,v in self.cq])*(1+self.ratio)
                if self.nt<self.nc:self.selected += [k for k,_ in list(self.cq)[1:1+self.nc-self.nt]]
                elif self.nt>self.nc:
                    q=list(self.sq)[:self.nt-self.nc+1];start=0
                    for i in reversed(range(len(q))):
                        if q[i][1]>self.threshold:start=i+1;break
                    self.selected += [k for k,_ in q[start+1:]]
            return
        if self.sq[0][0]<self.start:return
        if min(v for _,v in self.sq)>self.threshold:self.dead=True;return
        self.selected.append(self.sq[0][0])
    def final_epochs(self):
        if self.start is None:return [self.cq[-1][0]]
        selected=list(self.selected)
        if not self.dead:
            q=list(self.sq)[1:]
            for i,(epoch,_) in enumerate(q):
                if min(v for _,v in q[i:])>self.threshold:break
                selected.append(epoch)
        return sorted(selected or [self.start])
    def state_dict(self):return dict(nc=self.nc,nt=self.nt,ratio=self.ratio,cq=list(self.cq),sq=list(self.sq),start=self.start,threshold=float(self.threshold) if self.threshold is not None else None,dead=self.dead,selected=self.selected)
    def load_state_dict(self,s):
        self.__init__(s['nc'],s['nt'],s['ratio']);self.cq.extend(s['cq']);self.sq.extend(s['sq'])
        for key in ('start','threshold','dead','selected'):setattr(self,key,s[key])
