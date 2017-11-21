import numpy as np
import os
import gzip


def bin_data(x, mini, maxi, n_bins):
    """ 
    Bin data from continuous representation.
    
    :param x: real valued data in shape (time-points, batch)
    :param mini: minimum bin position
    :param maxi: maximum bin position
    :param n_bins: number of bins
    
    :return: binned data in shape (time-points, batch, bins)
    """
    out = np.zeros((x.shape[0], x.shape[1], n_bins))
    delta = (maxi - mini) / float(n_bins)
    for i, point in enumerate(np.arange(mini, maxi, delta)):
        out[:, :, i] = \
            np.logical_and(x >= point, x < point + delta).astype(float)

    return out


def unbin_data(x, mini, maxi, n_bins):
    """ 
    Convert binned data back to continuous representation.
    
    :param x: real valued data in shape (time-points, batch)
    :param mini: minimum bin position
    :param maxi: maximum bin position
    :param n_bins: number of bins

    :return: binned data in shape (time-points, batch, bins)
    """
    if np.any(np.isnan(x)):
        ix = np.where(np.logical_not(np.isnan(x[:, 0, 0])))[0]

        y = np.array([np.nan for _ in range(x.shape[0])])
        y[ix] = unbin_data(x[ix], mini, maxi, n_bins)
        return y
    else:
        estimate = np.where(x[:, 0, :])[1] * (maxi - mini) / float(n_bins) + mini

        return estimate


def load_mnist(path, kind='train'):
    """Load MNIST data

    :param path: path to data
    :param kind: 'train' or 'test'
    :return: (images, labels)
    """

    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def get_mnist(path):
    '''
    Get all MNIST data

    :param path: path to data
    :return: (train_images, train_labels, test_images, test_labels)
    '''
    train_images, train_labels = load_mnist(path)
    test_images, test_labels = load_mnist(path, "t10k")
    return train_images, train_labels, test_images, test_labels


class ToyData:
    """
    Simple Mackey-Glass time series toy-data.
    
    $\\frac{dx}{dt} = \\frac{\\beta \, x_{t - \\tau}}{1 + (x_{t - \\tau})^n} - \gamma \, x_t$
    
    See http://www.scholarpedia.org/article/Mackey-Glass_equation
    
    """

    def __init__(self, T=150, dt=1, beta=0.2, gamma=0.1, tau=17, n=10, n_bins=50):
        """
        :param T: max time to simulation
        :param dt: size of time-step for Euler discretization
        :param beta: parameter of equation
        :param gamma: parameter of equation
        :param tau: parameter of equation
        :param n: parameter of equation
        :param n_bins: number of bins to discretize output values
        
        """
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.n = n
        self.T = T
        self.dt = dt

        self.x = []
        self.n_bins = n_bins
        self.mini = 0.3
        self.maxi = 1.5

    def _step(self):

        delay = int(self.tau / self.dt)
        increment = self.dt* (self.beta * self.x[-delay] / (1 + self.x[-delay] ** self.n) \
                         - self.gamma * self.x[-1])
        self.x.append(self.x[-1] + increment)

    def simulate(self):
        """Simulate from model"""

        start = 0.4 + .8 * np.random.rand(1)[0]
        self.x = list(start * np.ones(int(self.tau / self.dt) + 1))
        for i in range(int(self.T / self.dt)):
            self._step()

        return np.array(self.x[int(self.tau / self.dt) + 1:])

    def _gen(self, batchsize):

        while True:

            batch = np.zeros((batchsize, self.T * self.dt))
            for i in range(batchsize):
                batch[i, :] = self.simulate()
            out = bin_data(batch.T, self.mini, self.maxi, self.n_bins)
            yield out, np.ones((out.shape[0], out.shape[1], self.n_bins))

    testing_gen = training_gen = _gen

    def get_readable(self, x):
        """Convert to plotting format"""
        return unbin_data(x, self.mini, self.maxi, self.n_bins)


class ToyDataMissing(ToyData):
    """ Simple Mackey-Glass time series toy-data with missing values.
    
    $\\frac{dx}{dt} = \\frac{\\beta \, x_{t - \\tau}}{1 + (x_{t - \\tau})^n} - \gamma \, x_t$
    
    See http://www.scholarpedia.org/article/Mackey-Glass_equation
    """

    def __init__(self, T=150, dt=1, beta=0.2, gamma=0.1, tau=17, n=10, n_bins=50, missing=0.1):
        """
        :param T: max time to simulation
        :param dt: size of time-step for Euler discretization
        :param beta: parameter of equation
        :param gamma: parameter of equation
        :param tau: parameter of equation
        :param n: parameter of equation
        :param n_bins: number of bins to discretize output values
        :param missing: percentage of missing-ness
        """

        ToyData.__init__(self, T, dt, beta=beta, gamma=gamma, tau=tau, n=n, n_bins=n_bins)
        self.missing = missing

    def _gen(self, batchsize):
        """
        
        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        while True:

            batch = np.zeros((self.T * self.dt, batchsize, self.n_bins))
            mask = np.zeros((batch.shape[0], batch.shape[1], self.n_bins))
            for i in range(batchsize):
                x_seq = self.simulate()
                ix = np.random.permutation(len(x_seq))
                nan_ix = ix[:int(self.missing * len(x_seq))]
                not_nan_ix = ix[int(self.missing * len(x_seq)):]
                binned = bin_data(x_seq[not_nan_ix, None],
                                        self.mini,
                                        self.maxi,
                                        self.n_bins)[:, 0, :]
                batch[not_nan_ix, i, :] = binned
                batch[nan_ix, i, :] = np.nan
                mask[not_nan_ix, i, :] = 1
            yield batch, mask

    testing_gen = training_gen = _gen


class CoupledToyData:

    def __init__(self, T=100, n_bins=100):
        """
        
        :param T: max time
        :param n_bins: number of discretization bins
        """

        self.T = T

        self.x = []
        self.n_bins = n_bins
        self.mini = -4
        self.maxi = 4

    def simulate(self):
        """simulate from model"""

        return np.random.multivariate_normal(np.zeros(2),
                                             np.array([[1, 0.8], [0.8, 1]]),
                                             size=self.T)

    def _gen(self, batchsize):
        """

        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        while True:

            batch = np.zeros((batchsize, self.T * 2))
            for i in range(batchsize):
                coupled = self.simulate()
                batch[i, 0::2] = coupled[:, 0]
                batch[i, 1::2] = coupled[:, 1]

            out = bin_data(batch.T, self.mini, self.maxi, self.n_bins)
            yield out, np.ones((out.shape[0], out.shape[1], self.n_bins))

    testing_gen = training_gen = _gen

    def get_readable(self, x):
        """Convert to plotting format"""
        out = np.zeros((x.shape[0] / 2, 2))
        out[:, 0] = unbin_data(x[::2, :, :], self.mini, self.maxi, self.n_bins)
        out[:, 1] = unbin_data(x[1::2, :, :], self.mini, self.maxi, self.n_bins)
        return out


class MNIST:
    """MNIST data modelled sequentially"""

    def __init__(self, path):
        """
        :param path: path to MNIST data
        """
        train_images, train_labels, test_images, test_labels = get_mnist(path)

        self.lookup = {0: 0, 1: 1}
        self.reverse_lookup = {0: 0, 1: 1}
        self.start = 0
        self.mini = -0.1
        self.maxi = 1.1
        self.n_bins = 2

        self.train_images = train_images / 255.
        self.train_labels = train_labels
        self.test_images = test_images / 255.
        self.test_labels = test_labels

    def _gen(self, batchsize, matrix):
        while True:
            images = matrix[np.random.permutation(len(matrix)), :]
            it = 0
            while it < len(images):
                out = self.discretize(images[it:it + batchsize])
                yield out, np.ones((out.shape[0], out.shape[1], self.n_bins))
                it += batchsize

    def discretize(self, matrix):
        """
        
        :param matrix: matrix to be binarized
        :return: binary_matrix
        """
        out = \
            np.concatenate((
                (matrix <= 0.5).T.astype(float)[:, :, None],
                (matrix > 0.5).T.astype(float)[:, :, None],
            ), axis=2)
        return out

    def training_gen(self, batchsize):
        """
        Training generator
        
        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        return self._gen(batchsize, self.train_images.reshape((60000, 784)))

    def testing_gen(self, batchsize):
        """
        Testing generator
        
        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        return self._gen(batchsize, self.test_images.reshape((10000, 784)))


class ShuffledMNIST(MNIST):
    """MNIST data modelled sequentially but shuffled in pixelization"""

    def __init__(self, path):
        """
        :param path: path to MNIST data
        """
        MNIST.__init__(self, path)
        np.random.seed(0)
        self.ix = np.random.permutation(self.train_images.shape[1])
        self.train_images = self.train_images[:, self.ix]
        self.test_images = self.test_images[:, self.ix]


class TimeSeries:
    """Time series data object taking raw time-series as input"""

    def __init__(self, x, window_length, mini, maxi, n_bins, train_T):
        """
        
        :param x: time-series
        :param window_length: length of prediction time-window for truncated BPP-TT
        :param mini: minimum value for binning
        :param maxi: maximum value for binning
        :param n_bins: number of bins
        :param train_T: time-value to truncate to for training (rest used for testing)
        """

        self.x = np.array(x)
        self.window_length = window_length
        self.mini = mini
        self.maxi = maxi
        self.n_bins = n_bins
        self.train_T = train_T

    def _gen(self, batchsize, x):
        while True:
            it = 0
            while it < len(x):
                batch = x[it: it + batchsize * self.window_length]
                try:
                    batch = batch.reshape((batchsize, self.window_length))
                    out = bin_data(batch.T, self.mini, self.maxi, self.n_bins)
                    yield out, np.ones(out.shape)
                except Exception:
                    break
                it += self.window_length * batchsize

    def training_gen(self, batchsize):
        """
        Training generator

        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        return self._gen(batchsize, self.x[:self.train_T])

    def testing_gen(self, batchsize):
        """
        Training generator

        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        return self._gen(batchsize, self.x[self.train_T:])


class NadeWrapper:
    """Wrapper to generators for NADE training"""

    def __init__(self, train_x, test_x, mini, maxi, n_bins):
        """
        
        :param train_x: real valued training data as numpy array in shape (n_samples, time_points)
        :param test_x: real valued training data as numpy array in shape (n_samples, time_points)
        :param mini: smallest bin
        :param maxi: largest bin
        :param n_bins: number of bins
        """
        self.train_x = train_x
        self.test_x = test_x
        self.mini = mini
        self.maxi = maxi
        self.n_bins = n_bins

    def _gen(self, batchsize, matrix):
        while True:
            temp = matrix[np.random.permutation(len(matrix)), :]
            it = 0
            while it < len(temp):
                out = bin_data(temp[it:it + batchsize], self.mini, self.maxi, self.n_bins)
                yield out, np.ones((out.shape[0], out.shape[1], self.n_bins))
                it += batchsize

    def training_gen(self, batchsize):
        """
        Training generator
    
        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        return self._gen(batchsize, self.train_x)

    def testing_gen(self, batchsize):
        """
        Testing generator
    
        :param batchsize: number of data points per batch
        :return: generator returning (batch, mask)
        """
        return self._gen(batchsize, self.test_x)