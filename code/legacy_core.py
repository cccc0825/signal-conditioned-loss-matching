"""Numerically unchanged kernel, prediction, and baseline routines."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist

def kernel(A,B,s,kind='se'):
    D=cdist(A,B,'sqeuclidean')/(A.shape[1]*s*s)
    if kind=='se':return np.exp(-.5*D)
    r=np.sqrt(3*np.maximum(D,0));return (1+r)*np.exp(-r)

def grouped_error(pred,Y,g):return float(np.mean([np.mean((pred[g==a]-Y[g==a])**2) for a in np.unique(g)]))

def fit_predict(K,V,Y,lam):
    A=K.copy();A.flat[::len(A)+1]+=len(A)*lam
    C=cho_solve(cho_factor(A,lower=True,check_finite=False),Y,check_finite=False)
    return V@C,C

def kcenter(X,m,seed):
    rng=np.random.default_rng(seed);v=int(rng.integers(len(X)));dist=np.full(len(X),np.inf);idx=[]
    for t in range(m):
        idx.append(v);dist=np.minimum(dist,np.sum((X-X[v])**2,axis=1));dist[idx]=-np.inf;v=int(np.argmax(dist))
    return np.array(idx)

def matching(H,m):
    mu=H.mean(1);diag=H.diagonal().copy();summed=np.zeros(len(H));used=np.zeros(len(H),bool);idx=[]
    for t in range(m):
        objective=diag-2*(t+1)*mu+2*summed;objective[used]=np.inf;v=int(np.argmin(objective));idx.append(v);used[v]=True;summed+=H[:,v]
    return np.array(idx)

def adaptive(K,Y,m,nugget,method,seed=0,initial=None):
    """Rank-one conditional updates; GEMSS criterion averaged over standardized joints."""
    n=len(K);J=Y.shape[1];L=np.zeros((n,m));pred=np.zeros_like(Y);variance=K.diagonal().copy()+nugget;sigsum=np.zeros(J);used=np.zeros(n,bool);idx=[]
    init=np.random.default_rng(seed).choice(n,2,replace=False) if initial is None else np.array(initial)
    for t in range(m):
        if method=='gemss_mo' and t<2:v=int(init[t])
        else:
            if method=='variance':score=variance.copy()
            else:
                sigma2=sigsum/t;post=variance[:,None]*sigma2[None,:];err=(Y-pred)**2
                c1=3*post**2/np.maximum(2*err**2+3*post**2,1e-300);score=np.mean(c1*err+(1-c1)*post,axis=1)
            score[used]=-np.inf;v=int(np.argmax(score))
        innovation=Y[v]-pred[v];denom=float(variance[v]);assert denom>0
        h=K[:,v].copy()
        if t:h-=L[:,:t]@L[v,:t]
        col=h/np.sqrt(denom);L[:,t]=col;pred+=col[:,None]*(innovation/np.sqrt(denom))[None,:]
        variance-=col**2;variance=np.maximum(variance,1e-14);sigsum+=innovation**2/denom;used[v]=True;idx.append(v)
    return np.array(idx)
