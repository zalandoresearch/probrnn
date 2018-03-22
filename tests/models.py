import unittest
from probrnn import models, data
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

params = \
    {
        "N_ITERATIONS": 100,
        "VALIDATE_EACH": 100,
        "SAVE_EACH": 1000,
        "LOG_EVERY": 50,
        "LEARNING_RATE": 0.01,
        "N_HIDDEN": 32,
        "N_BINS": 50,
        "BATCH_SIZE": 30,
        "WINDOW_LENGTH": 25
    }


def print_function(err, i, batch):
    if i % 1 == 0:
        print "err is {}; iteration is {}".format(err[-1], i)


class TestNADE(unittest.TestCase):
    def test_train(self):
        x = .5 * np.ones((1000, 20))
        datastruct = data.NadeWrapper(x, x, 0, 2, params["N_BINS"])

        model = models.NADE(datastruct, params=params)
        training = models.Training(model, "models/test_model", "models/test_model_training.json")

        training.train(lambda a, b, c: None)
        x = model.sample(10)
        self.assertTrue(len(np.unique(x)) == 1)


class TestTimeSeries(unittest.TestCase):
    def test_train(self):
        x = .5 * np.ones(10000)
        datastruct = data.TimeSeries(x, params["WINDOW_LENGTH"], 0, 2, params["N_BINS"], 9000)

        model = models.TimeSeriesPrediction(datastruct, params=params)
        training = models.Training(model, "models/test_model", "models/test_model_training.json")

        training.train(lambda a, b, c: None)
        x = model.sample(10)
        self.assertTrue(len(np.unique(x)) == 1)


if __name__ == "__main__":
    unittest.main()
