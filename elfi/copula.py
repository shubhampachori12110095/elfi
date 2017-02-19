import scipy.stats as ss
import scipy as sp
import numpy as np
import itertools

import matplotlib.pyplot as plt

from . import methods
from .distributions import Prior


def _full_cor_matrix(correlations, n):
    """Construct a full correlation matrix from pairwise correlations."""
    I = np.eye(n)
    O = np.zeros((n, n))
    indices = itertools.combinations(range(n), 2)
    for (i, inx) in enumerate(indices):
        O[inx] = correlations[i]

    return O + O.T + I


class Copula(object):
    """Sketch for the copula ABC method from: https://arxiv.org/abs/1504.04093

    Arguments
    ---------
    method: ABCMethod
      the ABC method used to approximate the marginal distributions
    parameter_dist: OrderedDict
      a dictionary that maps parameter nodes to their respective
      informative discrepancy nodes
    """

    def __init__(self, method, parameter_dist, **kwargs):
        self.arity = len(parameter_dist)
        self._log_pdf = None
        self._cm = None
        self.method = method
        self._parameter_dist = parameter_dist
        self._1d_methods = {}
        self._2d_methods = {}
        self.kdes = None
        self.estimated = False

    def estimate(self, n_samples=100):
        """Estimate the posterior using a Gaussian copula.

        Arguments
        ---------
        n_samples: int
          number of samples to use for each marginal estimate
        """
        self.kdes = self._marginal_kdes(n_samples)
        self._cm = self._cor_matrix(n_samples)
        self.estimated = True

    def sample(self, n_samples):
        """Sample values from the approzimate posterior."""
        raise NotImplementedError

    def diagnose_quality(self):
        """Evaluate whether the full posterior may be adequately modelled by
        a Gaussian copula."""
        raise NotImplementedError

    def _plot_marginal(self, inx, bounds, points=100):
        t = np.linspace(*bounds, points)
        return plt.plot(t, self.kdes[inx].pdf(t))

    def _construct_methods(self):
        """Constructs marginal ABC methods with default settings."""
        #TODO: is it possible to use samples directly instead?
        self._1d_methods = self._construct_1d_methods()
        self._2d_methods = self._construct_2d_methods()

    def _construct_1d_methods(self):
        methods = {k: self.method(distance_node=v, parameter_nodes=[k])
                   for k, v in self._parameter_dist.items()}
        return methods

    def _construct_2d_methods(self):
        pairs = itertools.combinations(self._parameter_dist, 2)
        #TODO: This is only a placeholder
        methods = {pair: self.method(distance_node=self._parameter_dist[pair[0]],
                                     parameter_nodes=list(pair))
                   for pair in pairs}
        return methods

    def _sample_from_marginal(self, marginal, n_samples):
        """Sample from the approximate marginal."""
        if isinstance(marginal, Prior):
            return self._sample_1d_marginal(marginal, n_samples)
        elif isinstance(marginal, tuple):
            return self._sample_2d_marginal(marginal, n_samples)

    def _sample_1d_marginal(self, marginal, n_samples):
        res = self._1d_methods[marginal].sample(n_samples=n_samples)
        return res.samples[marginal.name]

    def _sample_2d_marginal(self, marginal, n_samples):
        res = self._2d_methods[marginal].sample(n_samples=n_samples)
        sample = res.samples[marginal[0].name], res.samples[marginal[1].name]
        return sample
        
    def _marginal_kde(self, marginal, n_samples):
        #TODO: add 2d
        samples = self._sample_from_marginal(marginal, n_samples)
        kernel = ss.gaussian_kde(samples.reshape(-1))
        return kernel

    def _marginal_kdes(self, n_samples):
        marginal_params = self._parameter_dist
        kdes = [self._marginal_kde(m, n_samples) for m in marginal_params]
        return kdes

    def _eta_i(self, i, t):
        return ss.norm.ppf(self.kdes[i].integrate_box_1d(-np.inf, t))

    def _eta(self, theta):
        return np.array([self._eta_i(i, t) for (i, t) in enumerate(theta)])

    def _marginal_prod(self, theta):
        """Evaluate the logarithm of the poduct of the marginals."""
        res = 0
        for (i, t) in enumerate(theta):
            res += self.kdes[i].logpdf(t)
        return res
    
    def logpdf(self, theta):
        if len(theta.shape) == 1:
            return self._logpdf(theta)
        elif len(theta.shape) == 2:
            return np.array([self._logpdf(t) for t in theta])

    __call__ = logpdf

    def _logpdf(self, theta):
        n = self.arity
        if self.estimated:
            correlation_matrix = self._cm
            a = np.log(1/np.sqrt(np.linalg.det(correlation_matrix)))
            L = np.eye(n) - np.linalg.inv(correlation_matrix)
            quadratic = 1/2 * self._eta(theta).T.dot(L).dot(self._eta(theta))
            c = self._marginal_prod(theta)
            return a + quadratic + c
        else:
            raise ValueError("The marginal distributions"
                             " have not been estimated yet.")

    def _estimate_correlation(self, marginal, n_samples):
        samples = self._sample_from_marginal(marginal, n_samples)
        c1, c2 = samples[0].reshape(-1), samples[1].reshape(-1)
        r1 = np.argsort(c1) + 1
        r2 = np.argsort(c2) + 1
        n = len(r1)
        eta1 = ss.norm.ppf(r1/(n + 1))
        eta2 = ss.norm.ppf(r2/(n + 1))
        cor = np.corrcoef(eta1, eta2)[1,1]
        return cor

    def _cor_matrix(self, n_samples):
        """Construct an estimated correlation matrix."""
        pairs = itertools.combinations(self._parameter_dist, 2)
        correlations = [self._estimate_correlation(marginal, n_samples)
                        for marginal in pairs]
        cor = _full_cor_matrix(correlations, self.arity)
        return cor 
