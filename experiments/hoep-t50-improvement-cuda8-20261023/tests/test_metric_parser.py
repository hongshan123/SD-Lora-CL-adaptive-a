import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_job import parse_curve


class MetricParserTests(unittest.TestCase):
    def test_mixed_numpy_wrappers_and_plain_values(self):
        self.assertEqual(parse_curve('CNN top1 curve: [np.float64(92.11), 84.74, np.float32(7.1e1)]'),
                         [92.11,84.74,71.])

    def test_reject_expressions_and_nonfinite_values(self):
        for text in ['[float(7)]','[1+2]','[1e999]','[True]']:
            with self.assertRaises((ValueError,AssertionError)):
                parse_curve('CNN top1 curve: '+text)


if __name__=='__main__':unittest.main()
