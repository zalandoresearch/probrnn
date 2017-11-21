from probrnn import data
import numpy as np
from tqdm import tqdm


class NaiveSIS:
    """
    Simple sequential importance resampling with resampling at every time-step
    See https://en.wikipedia.org/wiki/Particle_filter
    """
    def __init__(self, model, x, n_particles=100, binned=True, quiet=False):
        """
        
        :param model: trained model 
        :param x: time-series with np.nan for missing data
        :param n_particles: number of particles
        :param binned: whether the input data is binned or not
        :param quiet: verbose output or not
        """
        self.model = model

        if not binned:
            self.nan_places = np.where(np.isnan(x))[0]
            self.places = np.where(np.logical_not(np.isnan(x)))[0]
            self.x = np.zeros((len(x), 1, model.data.n_bins))
            self.x[self.places, :, :] = data.bin_data(x[self.places, None],
                                                      model.data.mini,
                                                      model.data.maxi,
                                                      model.data.n_bins)
        else:
            self.x = x
            self.nan_places = np.where(np.isnan(x[:, 0, 0]))[0]
            self.places = np.where(np.logical_not(np.isnan(x[:, 0, 0])))[0]

        self.n_particles = n_particles
        self.quiet = quiet


    def impute(self):
        """
        :return: particles imputing self.x in binned format
        """
        particles = np.repeat(self.x, self.n_particles, axis=1)
        states = None

        if self.quiet:
            iterate = range(len(self.x))
        else:
            iterate = tqdm(range(len(self.x)))

        for i in iterate:

            if i == 0:
                predictions, states = self.model.graph.predict(None, states, batchsize=self.n_particles)
            else:
                predictions, states = self.model.graph.predict(particles[i - 1:i, :, :], states)

            if i in self.nan_places:
                for j in range(self.n_particles):
                    choice = np.random.choice(self.model.data.n_bins,
                                              p=predictions[-1, j, :])

                    particles[i, j, :] = 0
                    particles[i, j, choice] = 1
            else:
                if i == 0:
                    l = np.ones(self.n_particles) / float(self.n_particles)
                else:
                    l = predictions[-1, :, np.where(self.x[i, 0, :])[0][0]]

                w = l / np.sum(l)

                ix = np.random.choice(self.n_particles, size=self.n_particles, p=w)

                particles = particles[:, ix, :]
                w = w[ix]
        return particles

    def estimate(self):
        """
        :return: estimate of missing values
        """
        particles = self.impute()
        estimate = np.zeros(len(self.x))
        for i in range(len(self.x)):
            estimate[i] = \
                self.model.data.mini + \
                (self.model.data.maxi - self.model.data.mini) * \
                np.mean(np.where(particles[i, :, :])[1]) / float(self.model.data.n_bins)
        return estimate