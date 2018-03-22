import unittest
import numpy as np
from probrnn import data


class Test_bin_data(unittest.TestCase):

    def test(self):
        x = np.zeros((3, 2))
        x[:, 0] = [.2, .9, .1]
        x[:, 1] = [.99, .01, .3]

        binned = data.bin_data(x, 0, 1, 2)
        self.assertEqual(list(np.where(binned[:, 0, :])[1]), [0, 1, 0])
        self.assertEqual(list(np.where(binned[:, 1, :])[1]), [1, 0, 0])


class TestToyData(unittest.TestCase):
    def testToyData(self):
        pass


class TestNadeWrapper(unittest.TestCase):
    def test_gen(self):
        x = np.random.rand(100, 10)
        datastuct = data.NadeWrapper(x, x, 0, 1, 8)
        self.assertTrue(datastuct.training_gen(5).next()[0].shape, (10, 5, 8))
        self.assertTrue(datastuct.testing_gen(5).next()[0].shape, (10, 5, 8))



if __name__ == "__main__":
    unittest.main()
