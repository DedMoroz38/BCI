import numpy as np
import pytest
from replay_checks import assert_replay,assert_spec

def test_fp32_replay_accepts_observed_roundoff():
    p=np.array([[.6043633951329137,.20987996376215018,.18575664110493603]])
    q=np.array([[.6043574105635715,.20988166421845145,.18576092521797696]])
    assert assert_replay(p,q)<1e-5

def test_replay_still_rejects_changed_decisions_and_large_error():
    with pytest.raises(AssertionError):assert_replay([[.500001,.499999,0]],[[.499999,.500001,0]])
    with pytest.raises(AssertionError):assert_replay([[.6,.2,.2]],[[.59,.21,.2]])
    with pytest.raises(AssertionError):assert_spec({'w':np.array([1.0])},{'w':np.array([1.000001])})
    assert_spec({'w':np.array([1.0])},{'w':np.array([1.0])})
