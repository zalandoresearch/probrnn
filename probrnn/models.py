from probrnn import graphs, data
import json
from decimal import Decimal
import numpy as np


class fakefloat(float):
    def __init__(self, value):
        self._value = value
    def __repr__(self):
        return str(self._value)


def defaultencode(o):
    if isinstance(o, Decimal) or isinstance(o, np.float32):
        # Subclass float with custom repr?
        return fakefloat(o)
    raise TypeError(repr(o) + " is not JSON serializable")


class Training:
    """Training methods for RNNs"""

    def __init__(self, model, fn, log, imputer=None):
        """
        Backprop training routine for models.
        
        :param model: rnn model
        """
        self.model = model
        self.fn = fn
        self.log = log
        self.train_data = self.model.data.training_gen(model.params["BATCH_SIZE"])
        self.test_data = self.model.data.testing_gen(model.params["BATCH_SIZE"])
        self.imputer = imputer

    def train(self, display):
        """
        Train the model
        
        :param display: call back with signature: display(self.train_err, i, batch)
        """

        self.train_err = []
        self.train_it = []
        self.validation_err = []
        self.validation_it = []

        for i in range(self.model.params["N_ITERATIONS"]):
            batch, mask = self.train_data.next()

            if self.imputer:
                particle_batch = []
                particle_mask = []
                for j in range(batch.shape[1]):
                    particle_batch.append(self.imputer(self.model,
                                          batch[:,  j, :][:, None, :]).impute())
                    particle_mask.append(mask)

                batch = np.concatenate(tuple(particle_batch), 1)
                mask = np.concatenate(tuple(particle_mask), 1)

            self.train_err.append(self.model.graph.train_step((batch, mask)))
            self.train_it.append(i)

            if i % self.model.params["VALIDATE_EACH"] == 0:
                test_batch = self.test_data.next()
                self.validation_err.append(self.model.graph.test_error(test_batch))
                self.validation_it.append(i)

            if i % self.model.params["SAVE_EACH"] == 0:
                self.model.save(self.fn + "_" + str(i))

            if i % self.model.params["LOG_EVERY"] == 0:
                self.create_log()

            display(self.train_err, i, batch)

    def create_log(self):
        """Create a log file for training"""
        self.model.graph.get_stats()
        out = self.model.graph.summary
        out["training_error"] = zip(self.train_it, self.train_err)
        out["validation_error"] = zip(self.validation_it, self.validation_err)
        with open(self.log, "w") as f:
            f.write(json.dumps(out, default=defaultencode))


class BaseModel:
    """Base model class for rnn sequential models."""

    def __init__(self, graph, data, params=None, fn=None):
        """
        
        :param params: 
        :param fn: 
        """

        self.data = data
        self.fn = fn
        self.params = params

        if self.fn is None:
            self.graph = graph(self.params)
            self.graph.initialize()
        else:
            print "restoring..."
            with open(self.fn + ".json") as f:
                self.params = json.load(f)
            self.graph = graph(self.params)
            self.graph.load(self.fn)

    def save(self, filename):
        """Save graph and parameters
        
        :param filename: path to saved files
        """
        self.graph.save(filename)
        with open(filename + ".json", "w") as f:
            f.write(json.dumps(self.params))


class NADE(BaseModel):
    """Neural autoregressive density estimation"""

    def __init__(self, data, params=None, fn=None):
        """
        
        :param params: dictionary of parameters passed to graphs.py and Training
        :param fn: filename if loading from disk
        """
        BaseModel.__init__(self, graphs.NADE, data, params=params, fn=fn)

    def sample(self, n):
        """Sample n-time steps"""
        vec = None
        states = None
        out = []
        for i in range(n):
            p, states = self.graph.predict(vec, states)
            vec = np.zeros((1, 1, self.params["N_BINS"]))
            choice = np.random.choice(range(self.params["N_BINS"]),
                                      p=np.squeeze(p))
            vec[0, 0, choice] = 1
            out.append(choice)
        return np.array(out)


class TimeSeriesPrediction(BaseModel):
    """Time series prediction with RNNs"""

    def __init__(self, data, params=None, fn=None):
        """

        :param params: dictionary of parameters passed to graphs.py and Training
        :param fn: filename if loading from disk
        """
        BaseModel.__init__(self, graphs.TimeSeriesPrediction, data, params=params, fn=fn)

    def sample(self, n):
        """Sample n-time steps"""
        predicted, _ = self.data.training_gen(1).next()
        out = []
        for i in range(n):
            p, _ = self.graph.predict(predicted[-self.params["WINDOW_LENGTH"] + 1:], None)
            vec = np.zeros((1, 1, self.params["N_BINS"]))
            choice = np.random.choice(range(self.params["N_BINS"]),
                                      p=np.squeeze(p))
            vec[0, 0, choice] = 1
            predicted = np.concatenate((predicted, vec), 0)
            out.append(choice)
        return np.array(out)

