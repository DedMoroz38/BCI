import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
try:
    from reproduce_synthetic import pack, unpack, matrix_function, subset_features, eta
except ImportError:
    pack = None

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(pack, 'Synthetic experiment implementation is missing')

    def test_vectorization_preserves_frobenius_norm(self):
        a = np.array([[1., 2.], [2., 3.]])
        self.assertAlmostEqual(np.linalg.norm(pack(a)), np.linalg.norm(a))
        np.testing.assert_allclose(unpack(pack(a), 2), a)

    def test_features_are_exact_submatrix_logarithms(self):
        s = np.array([[.1, .4], [.4, -.2]])
        r = matrix_function(s, np.exp)
        x = subset_features(r[None], np.array([0]))
        self.assertAlmostEqual(x[0, 0], np.log(r[0, 0]))
        self.assertGreater(abs(x[0, 0] - s[0, 0]), .01)

    def test_overlap_matches_enumeration(self):
        from itertools import combinations
        subsets = [set(s) for s in combinations(range(5), 3)]
        measured = np.mean([len(a & b) * (len(a & b) + 1) / 2 / 6
                            for a in subsets for b in subsets])
        self.assertAlmostEqual(eta(5, 3), measured)
        self.assertAlmostEqual(eta(16, 16), 1.)
        self.assertAlmostEqual(eta(16, 1), 1/16)

if __name__ == '__main__':
    unittest.main()
