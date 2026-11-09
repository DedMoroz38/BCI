"""Causal prefix Gram-PCA/soft-LDA, adapted from DorinDaniil (MIT)."""
import numpy as np
from scipy.linalg import eigh

def spectral(x, kind):
    w, v = np.linalg.eigh((x + np.swapaxes(x, -1, -2)) / 2)
    if kind == 'log':
        floor = np.maximum(np.max(w, axis=-1, keepdims=True)*1e-12, 1e-20)
        w = np.log(np.maximum(w, floor))
    elif kind == 'exp':
        w = np.exp(np.clip(w, -60, 60))
    return (v*w[..., None, :]) @ np.swapaxes(v, -1, -2)

def upper(x):
    i, j = np.triu_indices(x.shape[-1])
    return x[..., i, j] * np.where(i == j, 1., np.sqrt(2))

def soft_assignment(logits, tau, iterations=30):
    """Exact row masses; KL-relaxed column scaling (tau=1: author Sinkhorn)."""
    K = np.exp(np.clip(logits-logits.max(-1, keepdims=True), -230, 0))
    v = np.ones((len(K), 3)); mass = K.shape[1]/3
    for _ in range(iterations):
        u = 1/np.maximum(np.einsum('bnc,bc->bn', K, v), 1e-100)
        col = np.einsum('bnc,bn->bc', K, u)
        v = np.clip((mass/np.maximum(col, 1e-100))**tau[:, None], 1e-60, 1e60)
    q = K*v[:, None, :]
    return q/q.sum(-1, keepdims=True)

def solve_many(P, log_prior, cases):
    """Batch independent hyperparameters/priors, sharing only label-free PCA."""
    b, n, _ = log_prior.shape; k = P.shape[1]
    tau = np.array([c['tau'] for c in cases])
    beta = np.array([c['beta'] for c in cases])
    shrink = np.array([c['shrink'] for c in cases])
    q = soft_assignment(log_prior, tau)
    for _ in range(5):
        mu = np.einsum('bnc,nd->bcd', q, P)/np.maximum(q.sum(1)[..., None], 1e-12)
        residual = P[None]-q@mu
        S = np.swapaxes(residual, -1, -2)@residual/n
        tr = np.trace(S, axis1=-2, axis2=-1)/k
        S = (1-shrink[:, None, None])*S + (shrink*tr+1e-10)[:, None, None]*np.eye(k)
        inv_mu = np.linalg.solve(S, np.swapaxes(mu, -1, -2))
        ll = P[None]@inv_mu-.5*np.einsum('bcd,bdc->bc', mu, inv_mu)[:, None, :]
        q = soft_assignment(log_prior+beta[:, None, None]*ll, tau)
    return q[:, -1]

def features(logcov, source, geometry, priors):
    """Each row uses source stats and current/past target inputs only.

    Pseudo reference uses frozen Lag4 probabilities, never target labels.
    Each class has its own soft log-cov mean; their equal-weight mean defines M.
    """
    mode = geometry.split('_', 1)[1]
    if mode == 'log': return upper(logcov).reshape(len(logcov), -1)
    if mode == 'fixed':
        W = spectral(-.5*source, 'exp')
        return upper(spectral(W@spectral(logcov, 'exp')@W, 'log')).reshape(len(logcov), -1)
    weight = 20. if mode == 'running20' else 5.
    total = weight*source.copy(); count = weight
    class_total = np.repeat((weight/3*source)[None], 3, axis=0)
    class_count = np.full(3, weight/3)
    out = []
    for i, L in enumerate(logcov):
        if mode.startswith('pseudo'):
            p = priors[i].copy()
            if mode == 'pseudo_sharp': p = p*p; p /= p.sum()
            class_total += p[:, None, None, None]*L[None]
            class_count += p
            anchor = np.mean(class_total/class_count[:, None, None, None], axis=0)
        else:
            total += L; count += 1; anchor = total/count
        W = spectral(-.5*anchor, 'exp')
        out.append(upper(spectral(W@spectral(L, 'exp')@W, 'log')).ravel())
    return np.asarray(out)

def gram_for_geometry(logs, sources, geometry, priors):
    if geometry.startswith('joint_'):
        mode = geometry.split('_', 1)[1]
        za = features(logs[0], sources[0], 'ordinary_'+mode, priors)
        zb = features(logs[1], sources[1], 'lag_'+mode, priors)
        ga, gb = za@za.T, zb@zb.T
        # Per-prefix trace normalization below, so no future scale leaks.
        return ga, gb
    r = int(geometry.startswith('lag_'))
    z = features(logs[r], sources[r], geometry, priors)
    return (z@z.T,)

def balance(p):
    m = (10/3+np.cumsum(p, axis=0))/(10+np.arange(1, len(p)+1))[:, None]
    q = p/m
    return q/q.sum(1, keepdims=True)

def causal_geometry(grams, priors, cases, deadline=None):
    import time
    n_total = len(priors[0]); fallback = [balance(p) for p in priors]
    out = np.stack([np.stack(fallback) for _ in cases])  # cases,2,n,3
    for n in range(20, n_total+1):
        if deadline is not None and n % 10 == 0 and time.monotonic() > deadline:
            raise TimeoutError('Wall budget reached; completed fold artifacts retained')
        if len(grams) == 1: g = grams[0][:n, :n]
        else:
            gs = [v[:n, :n] for v in grams]
            g = sum(v/max(np.trace(v)/n, 1e-12) for v in gs)/len(gs)
        centered = g-g.mean(0)[None]-g.mean(1)[:, None]+g.mean()
        active = [i for i, c in enumerate(cases) if n >= c['warm']]
        for k0 in sorted({cases[i]['k'] for i in active}):
            ids = [i for i in active if cases[i]['k'] == k0]
            k = max(1, min(k0, n//3))
            _, vec = eigh(centered, subset_by_index=[n-k, n-1], check_finite=False)
            P = vec[:, ::-1]*np.sqrt(n)
            batch_cases = [cases[i] for i in ids for _ in priors]
            lp = np.stack([np.log(np.maximum(p[:n], 1e-9)) for i in ids for p in priors])
            qs = solve_many(P, lp, batch_cases).reshape(len(ids), len(priors), 3)
            for j, i in enumerate(ids):
                c = cases[i]; ramp = min(1., (n-c['warm']+1)/c['ramp']) if c['ramp'] else 1.
                a = c['strength']*ramp
                out[i, :, n-1] = a*qs[j]+(1-a)*np.stack([p[n-1] for p in fallback])
    return out

def stability_predictions(a, b, fallback):
    mean = (a+b)/2
    js = .5*np.sum(a*np.log(np.maximum(a, 1e-15)/np.maximum(mean, 1e-15))+
                     b*np.log(np.maximum(b, 1e-15)/np.maximum(mean, 1e-15)), axis=-1)
    out = {'views_mean': mean, 'views_mean_half': .5*mean+.5*fallback}
    for scale, name in ((.05, 'views_stable005'), (.2, 'views_stable02')):
        reliability = np.exp(-js/scale)[..., None]
        out[name] = reliability*mean+(1-reliability)*fallback
    return out
