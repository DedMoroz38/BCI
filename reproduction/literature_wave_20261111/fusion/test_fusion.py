import numpy as np
import pytest
from fusion_wave import validate_rows,source_statistics,fixed_geometry_predictions

def test_wrong_forecast_row_order_is_rejected():
    expected={k:np.array([1,2,3]) for k in ('y','subject','session','context','run','onset')}
    actual={k:v.copy() for k,v in expected.items()}
    actual['onset']=actual['onset'][::-1]
    with pytest.raises(ValueError,match='onset'):
        validate_rows(actual,expected)

def test_target_changes_cannot_change_source_statistics():
    logs=np.arange(6*2*2).reshape(6,2,2).astype(float)
    reference=source_statistics([logs],np.array([0,2,4]))
    changed=logs.copy();changed[[1,3,5]]=1e10
    np.testing.assert_equal(source_statistics([changed],np.array([0,2,4])),reference)

def test_predictions_use_only_each_natural_prefix():
    rng=np.random.default_rng(4)
    logs=rng.normal(size=(25,2,2,2));logs=(logs+logs.swapaxes(-1,-2))*.05
    source=np.zeros((2,2,2))
    priors=rng.dirichlet(np.ones(3),size=(2,25))
    expected=fixed_geometry_predictions(logs,source,priors)
    changed=logs.copy();changed[22:]*=100
    changed_priors=priors.copy();changed_priors[:,22:]=[1,0,0]
    actual=fixed_geometry_predictions(changed,source,changed_priors)
    np.testing.assert_allclose(actual[:,:22],expected[:,:22],atol=1e-10)
    np.testing.assert_allclose(expected.sum(-1),1,atol=1e-10)
