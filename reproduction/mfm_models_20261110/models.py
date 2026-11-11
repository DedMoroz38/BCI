"""Weighted PLS/HOPLS2, HOSVD and structured logistic regression."""
import numpy as np
from scipy.special import softmax,logsumexp
from sklearn.linear_model import LogisticRegression
from core import upper,unupper

class Scaler:
    def fit(self,x,w):
        self.mean=np.average(x,axis=0,weights=w)
        self.std=np.sqrt(np.average((x-self.mean)**2,axis=0,weights=w))
        self.std=np.maximum(self.std,1e-6);return self
    def transform(self,x):return (x-self.mean)/self.std

def classifier(a,b,y,w,C=.1):
    s=Scaler().fit(a,w);a=s.transform(a);b=s.transform(b)
    clf=LogisticRegression(C=C,max_iter=300,solver='lbfgs')
    clf.fit(a,y,sample_weight=w)
    return clf.predict_proba(b),dict(scaler=s,clf=clf)

def mode_dot(a,matrix,axis):
    return np.moveaxis(np.tensordot(matrix,a,axes=(1,axis)),0,axis)

def project(a,factors,start=0):
    for j,u in enumerate(factors):a=mode_dot(a,u.T,start+j)
    return a

def expand(a,factors):
    for j,u in enumerate(factors):a=mode_dot(a,u,j)
    return a

def basis(a,axis,rank):
    x=np.moveaxis(a,axis,0).reshape(a.shape[axis],-1)
    gram=x@x.T;_,v=np.linalg.eigh((gram+gram.T)/2)
    return v[:,-min(rank,len(v)):][:,::-1]

def hooi(a,ranks,iterations=15):
    u=[basis(a,j,r) for j,r in enumerate(ranks)];old=-1
    for _ in range(iterations):
        for j,r in enumerate(ranks):
            q=a
            for k,v in enumerate(u):
                if k!=j:q=mode_dot(q,v.T,k)
            u[j]=basis(q,j,r)
        norm=float(np.sum(project(a,u)**2))
        if abs(norm-old)<1e-7*max(norm,1):break
        old=norm
    return project(a,u),u

class HOSVD:
    def __init__(self,ranks):self.ranks=tuple(ranks)
    def fit(self,x,y,w):
        self.mean=np.average(x,axis=0,weights=w).astype('float32')
        z=(x-self.mean)*np.sqrt(w)[:,None,None,None]
        self.factors=[basis(z,j+1,r) for j,r in enumerate(self.ranks)]
        return self
    def transform(self,x):return project(x-self.mean,self.factors,1).reshape(len(x),-1)

class WeightedPLS:
    def __init__(self,components=4):self.components=components
    def fit(self,x,y,w):
        x=x.reshape(len(x),-1);self.mean=np.average(x,axis=0,weights=w)
        yy=np.eye(3)[y];yy-=np.average(yy,axis=0,weights=w)
        E=((x-self.mean)*np.sqrt(w)[:,None]).astype('float64');F=yy*np.sqrt(w)[:,None]
        weights=[];loads=[]
        for _ in range(self.components):
            C=E.T@F;u,_,_=np.linalg.svd(C,full_matrices=False);v=u[:,0]
            t=E@v;den=float(t@t)
            if den<1e-15:break
            p=E.T@t/den;q=F.T@t/den
            E-=t[:,None]*p;F-=t[:,None]*q;weights.append(v);loads.append(p)
        W=np.stack(weights,axis=1);P=np.stack(loads,axis=1)
        self.rotation=W@np.linalg.pinv(P.T@W);return self
    def transform(self,x):return (x.reshape(len(x),-1)-self.mean)@self.rotation

class HOPLS2:
    """Paper Algorithm2, matrix response; projection Eq27, classification adapter.

    Source weighted centering, HOOI ranks(1,*ranks), cross-core latent vector,
    normalized scores, X/Y deflation. No target fitting. CE head is separate.
    """
    def __init__(self,components=4,ranks=(4,4,2)):
        self.components=components;self.ranks=tuple(ranks)
    def fit(self,x,y,w):
        self.mean=np.average(x,axis=0,weights=w).astype('float32')
        yy=np.eye(3)[y];yy-=np.average(yy,axis=0,weights=w)
        shape=(len(w),)+(1,)*(x.ndim-1)
        E=((x-self.mean)*np.sqrt(w).reshape(shape)).astype('float32');F=(yy*np.sqrt(w)[:,None]).astype('float32')
        self.filters=[];self.loadings=[];self.response=[]
        for _ in range(self.components):
            C=np.tensordot(F.T,E,axes=(1,0))
            gc,factors=hooi(C,(1,)+self.ranks)
            q=factors[0][:,0];P=factors[1:];flat=gc.reshape(-1)
            projected=project(E,P,1).reshape(len(E),-1)
            t=projected@flat/max(float(flat@flat),1e-15)
            norm=np.linalg.norm(t)
            if norm<1e-12:break
            t=t/norm
            gx=(t@projected).reshape(gc.shape[1:]);loading=expand(gx,P).astype('float32')
            gnorm=float(np.sum(gx*gx))
            if gnorm<1e-15:break
            self.filters.append((loading/gnorm).reshape(-1))
            self.loadings.append(loading);d=float(t@(F@q));self.response.append(d*q)
            E-=t.reshape(shape)*loading;F-=t[:,None]*(d*q)
        if not self.filters:raise FloatingPointError('HOPLS has no valid component')
        self.rotation=np.stack(self.filters,axis=1);return self
    def transform(self,x):return (x-self.mean).reshape(len(x),-1)@self.rotation

class StructuredLR:
    """Symmetric-matrix nuclear or hard rank prox with multinomial CE.

    Features use sqrt2 upper vectorization; symmetric reconstruction is an
    isometry, so Frobenius penalty equals ordinary vector L2.
    """
    def __init__(self,dim,nuclear=0.,rank=None,iterations=60):
        self.dim=dim;self.nuclear=nuclear;self.rank=rank;self.iterations=iterations
    def fit(self,x,y,w,initial,C=.01):
        w=w/w.sum();lambda2=1/(C*initial['weight_sum'])
        z=np.column_stack([x,np.ones(len(x))]);target=np.eye(3)[y]
        A=np.column_stack([initial['coef'],initial['intercept']]).astype('float64')
        def project_weights(B,step):
            mats=unupper(B[:,:-1],self.dim);ev,U=np.linalg.eigh(mats)
            if self.rank is not None:
                keep=np.argsort(np.abs(ev),axis=-1)[:,-self.rank:]
                mask=np.zeros_like(ev);np.put_along_axis(mask,keep,1,axis=-1);ev*=mask
            else:ev=np.sign(ev)*np.maximum(np.abs(ev)-step*self.nuclear,0)
            B=B.copy();B[:,:-1]=upper((U*ev[:,None,:])@U.swapaxes(-1,-2));return B
        def smooth(B,gradient=False):
            logits=z@B.T
            value=float(np.sum(w*(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y]))+.5*lambda2*np.sum(B[:,:-1]**2))
            if not gradient:return value
            g=((softmax(logits,axis=1)-target)*w[:,None]).T@z
            g[:,:-1]+=lambda2*B[:,:-1];return value,g
        # Deterministic power estimate followed by backtracking, not a gate.
        v=np.ones(z.shape[1]);v/=np.linalg.norm(v)
        for _ in range(8):
            v=z.T@(w*(z@v));v/=max(np.linalg.norm(v),1e-15)
        L=max(.5*float(np.sum(w*(z@v)**2))+lambda2,1e-6)
        A=project_weights(A,0);B=A.copy();moment=1;history=[]
        for it in range(self.iterations):
            loss,g=smooth(B,True)
            for _ in range(15):
                new=project_weights(B-g/L,1/L);diff=new-B
                if smooth(new)<=loss+np.sum(g*diff)+.5*L*np.sum(diff*diff)+1e-12:break
                L*=2
            history.append(smooth(new))
            change=np.linalg.norm(new-A)/max(np.linalg.norm(A),1)
            if self.rank is None:
                next_moment=(1+np.sqrt(1+4*moment*moment))/2
                B=new+(moment-1)/next_moment*(new-A);moment=next_moment
            else:B=new.copy()
            A=new
            if change<1e-5:break
        self.coef=A[:,:-1];self.intercept=A[:,-1];self.history=history
        return self
    def predict_proba(self,x):return softmax(x@self.coef.T+self.intercept,axis=1)
