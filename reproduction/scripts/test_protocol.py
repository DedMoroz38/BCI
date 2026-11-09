"""Checks that protect the paper protocol from split leakage or ordering changes."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    from reproduce_eeg import validate_splits
except ImportError:
    validate_splits = None


def windows(subjects, n):
    return SimpleNamespace(
        X=np.zeros((n, 41, 480), dtype=np.float32), y=np.arange(n) % 3,
        subject=np.array([subjects[i % len(subjects)] for i in range(n)]),
        session=np.full(n, '01'), context=np.full(n, 'Graz'),
        run=np.ones(n, int), onset=np.arange(n, dtype=float),
        ch_names=np.array([str(i) for i in range(41)]))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validate_splits, 'EEG runner with protocol validation is missing')
        self.parts = {s: windows(ids, len(ids) * 3) for s, ids in [
            ('train', ['a', 'b']), ('val', ['c']), ('test', ['d'])]}
        for w in self.parts.values():
            order = np.lexsort((w.onset, w.run, w.session, w.subject))
            for k in ['X', 'y', 'subject', 'session', 'context', 'run', 'onset']:
                setattr(w, k, getattr(w, k)[order])

    def test_accepts_disjoint_ordered_data(self):
        counts = validate_splits(self.parts, check_paper_counts=False)
        self.assertEqual(counts['train']['subjects'], 2)

    def test_rejects_subject_leakage(self):
        self.parts['test'].subject[:] = 'a'
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_splits(self.parts, check_paper_counts=False)

    def test_rejects_paper_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'paper counts'):
            validate_splits(self.parts)

    def test_rejects_reversed_stream(self):
        self.parts['val'].onset = self.parts['val'].onset[::-1]
        with self.assertRaisesRegex(ValueError, 'order'):
            validate_splits(self.parts, check_paper_counts=False)

    def test_rejects_missing_classes(self):
        self.parts['val'].y[:] = 0
        with self.assertRaisesRegex(ValueError, 'classes'):
            validate_splits(self.parts, check_paper_counts=False)


if __name__ == '__main__':
    unittest.main()
