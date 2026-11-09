"""Fixed exploratory grid; no selection or label-dependent settings at runtime."""
GEOMETRIES = (
    'ordinary_running', 'lag_running', 'ordinary_fixed', 'lag_fixed',
    'ordinary_log', 'lag_log', 'ordinary_pseudo', 'lag_pseudo',
    'lag_pseudo_sharp', 'ordinary_running20', 'lag_running20',
    'joint_running', 'joint_fixed',
)

def configurations():
    settings = [
        ('hard', dict(tau=1.)), ('soft25', dict(tau=.25)),
        ('soft50', dict(tau=.5)), ('soft75', dict(tau=.75)),
        ('no_marginal', dict(tau=0.)),
        ('soft50_beta05', dict(tau=.5, beta=.5)),
        ('soft50_beta2', dict(tau=.5, beta=2.)),
        ('soft50_ramp60', dict(tau=.5, ramp=60)),
    ]
    extra = [
        ('soft50_k10', dict(tau=.5, k=10)),
        ('soft50_k30', dict(tau=.5, k=30)),
        ('soft50_shrink02', dict(tau=.5, shrink=.2)),
        ('soft50_shrink08', dict(tau=.5, shrink=.8)),
        ('soft50_warm50', dict(tau=.5, warm=50)),
        ('soft50_warm80', dict(tau=.5, warm=80)),
        ('soft50_half', dict(tau=.5, strength=.5)),
    ]
    out = []
    for geometry in GEOMETRIES:
        for name, update in settings + (extra if geometry in ('lag_running', 'joint_running') else []):
            c = dict(name=geometry+'__'+name, geometry=geometry, tau=1., beta=1.,
                     k=20, iters=5, shrink=.5, warm=20, ramp=0, strength=1.)
            c.update(update); out.append(c)
    return out

CONFIGS = configurations()
STABILITY = ('views_mean', 'views_stable005', 'views_stable02', 'views_mean_half')
