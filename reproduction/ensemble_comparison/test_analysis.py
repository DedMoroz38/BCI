import pytest
import analyze

def test_pairing_is_by_person_not_row_order():
    rows=[{'tag':'loso_P01','variant':'Lag4','session_mean':.5},
          {'tag':'loso_P02','variant':'Lag4','session_mean':.7},
          {'tag':'loso_P02','variant':'all','session_mean':.8},
          {'tag':'loso_P01','variant':'all','session_mean':.4}]
    r=analyze.paired(rows,'all')
    assert r['mean_delta']==pytest.approx(0)
    assert r['wins']==1 and r['n']==2
    with pytest.raises(ValueError):analyze.paired(rows[:-1],'all')
