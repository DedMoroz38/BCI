"""FP32 replay validation without requiring accidental bitwise identity."""
import numpy as np

def assert_replay(expected,actual,atol=1e-4):
    expected=np.asarray(expected);actual=np.asarray(actual)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=atol)
    np.testing.assert_array_equal(actual.argmax(-1),expected.argmax(-1))
    return float(np.max(np.abs(actual-expected)))

def assert_spec(expected,actual):
    if isinstance(expected,dict):
        assert expected.keys()==actual.keys()
        for key in expected:assert_spec(expected[key],actual[key])
    elif isinstance(expected,(list,tuple)):
        assert len(expected)==len(actual)
        for x,y in zip(expected,actual):assert_spec(x,y)
    elif isinstance(expected,np.ndarray):np.testing.assert_array_equal(expected,actual)
    else:assert expected==actual
