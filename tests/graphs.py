import unittest
from probrnn import graphs
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

params = {
    "LEARNING_RATE": 0.0001,
    "N_HIDDEN": 64,
    "N_BINS": 2,
    "BATCH_SIZE": 30,
    "WINDOW_LENGTH": 23,
}


class TestNADE(unittest.TestCase):

    def test_initialize(self):
        graph = graphs.NADE(params)
        graph.initialize()

    def test_get_stats(self):
        graph = graphs.NADE(params)
        graph.initialize()
        d = graph.get_stats()
        d = graph.get_stats()
        self.assertTrue(len(graph.summary.keys()) > 5)


class TestTimeSeriesPrediction(unittest.TestCase):

    def test_train_step(self):
        graph = graphs.TimeSeriesPrediction(params)
        graph.initialize()
        x = np.random.randn(params["WINDOW_LENGTH"], params["BATCH_SIZE"], params["N_BINS"])
        graph.train_step((x, None))

    def test_test_error(self):
        graph = graphs.TimeSeriesPrediction(params)
        graph.initialize()
        x = np.random.randn(params["WINDOW_LENGTH"], params["BATCH_SIZE"], params["N_BINS"])
        graph.test_error((x, None))

if __name__ == "__main__":
    unittest.main()