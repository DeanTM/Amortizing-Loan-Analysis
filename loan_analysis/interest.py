from typing import Sequence
import numpy as np
from scipy.optimize import fmin

from abc import ABC, abstractmethod


def get_autocorrelation(trace, maxlag=30):
    autocorrelations = np.zeros(maxlag)
    autocorrelations[0] = 1.
    for i in range(1,maxlag):
        corrcoef = np.corrcoef(trace[:-i], trace[i:])[0,1]
        autocorrelations[i] = corrcoef
    return autocorrelations

def gaussian_func(x, tau):
    return np.exp(-(x**2)*tau)

def laplace_func(x, tau):
    return np.exp(-np.abs(x)*tau)

def mixture_func(x, params):
    tau0, tau1, p = params
    p = 1./(1.+np.exp(-p))
    return p*gaussian_func(x, tau0) + (1-p)*laplace_func(x, tau1)

def get_loan_rates(
    interest_rate_curve: Sequence[float],
    spread: float=0.0,
    renew_every: int=1
    ) -> np.ndarray:
    return np.repeat(
        interest_rate_curve[::renew_every],
        renew_every, axis=0) + spread


class IBORProcess(ABC):
    """Abstract base class for simulators of interbank offered rates."""
    @abstractmethod
    def sample_ibor_curve(
        self,
        number_of_months,
        num_samples=1,
        initial_value=None
        ) -> np.ndarray:
        raise NotImplementedError("This method needs to be overwritten")


class IBORGaussianProccess(IBORProcess):
    tau = None
    SIGMA = None
    L_cholesky = None
    LAMBDA = None

    def __init__(
        self,
        historical_ibor,  # give in fractions not percentages
        autocorrelation_maxlag=100,
        kernel_type='laplace',
        fit=True
        ):
        self.historical_ibor = historical_ibor
        self.autocorrelation_maxlag = autocorrelation_maxlag
        self.kernel_type = kernel_type
        self.autocorrelations = get_autocorrelation(
            trace = self.historical_ibor,
            maxlag = self.autocorrelation_maxlag)
        self.standard_deviation = np.std(self.historical_ibor)
        if fit:
            self.refit_kernel(suppress=True)
    
    def get_count_per_autocorrelation(self):
        weights = self.historical_ibor.shape[0] \
            - np.arange(self.historical_ibor.shape[0])
        return weights[:self.autocorrelation_maxlag]

    def compute_observed_autocorrelation(self, max_lag):
        autocorrelations = np.zeros(max_lag)
        autocorrelations[0] = 1.
        for i in range(1,max_lag):
            corrcoef = np.corrcoef(
                self.historical_ibor[:-i],
                self.historical_ibor[i:])[0,1]
            autocorrelations[i] = corrcoef
        return autocorrelations

    def kernel_func(self, x, tau=None):
        if tau is None:
            assert self.tau is not None
            tau = self.tau
        if self.kernel_type.lower() == 'gaussian':
            return gaussian_func(x, tau)
        elif self.kernel_type.lower() == 'laplace':
            return laplace_func(x, tau)
        elif self.kernel_type.lower() == 'mixture':
            return mixture_func(x, tau)
        else:
            raise NotImplementedError(f"Kernel {self.kernel} not implemented.")
    
    def kernel(self, i,j):
        """Computes inferred covariance"""
        assert self.tau is not None, "Must fit first."
        return self.kernel_func(np.abs(i-j))

    def populate_covariance(self, size=100):
        self.LAMBDA = None  # reset precision matrix
        self.SIGMA = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                self.SIGMA[i,j] = self.kernel(i,j) *self.standard_deviation**2
        # check positive definite
        EIGS, EIGVECS = np.linalg.eigh(self.SIGMA)
        if EIGS.min() < 1e-12:
            EIGS[EIGS < 1e-12] = 1e-12
            self.SIGMA = (EIGVECS @ np.diag(EIGS) @ EIGVECS.T)
        self.LAMBDA = np.linalg.inv(self.SIGMA)
        return None

    def sample_ibor_curve(
        self,
        number_of_months=None,
        num_samples=1,
        initial_value=None
        ):
        refit_all = False
        if number_of_months is None and self.SIGMA is not None:
            number_of_months = self.SIGMA.shape[0]
        elif self.SIGMA is None:
            refit_all = True
        elif self.SIGMA.shape[0] != number_of_months:
            refit_all = True
        if refit_all:
            self.populate_covariance(size=number_of_months)
            self.L_cholesky = np.linalg.cholesky(self.SIGMA)
        elif self.L_cholesky is None:
            self.L_cholesky = np.linalg.cholesky(self.SIGMA)
        rates_uncorrelated = np.random.randn(number_of_months,num_samples)
        rates_correlated = self.L_cholesky @ rates_uncorrelated
        if initial_value is not None:
            rates_correlated += initial_value - rates_correlated[0,:]
        return rates_correlated
    
    def sample_conditional_ibor_curve(
        self, number_of_months_total,
        given_rates,
        given_timepoints=None,
        num_samples=1
    ):
        # following https://web.ics.purdue.edu/~ibilion/www.zabaras.com/Courses/BayesianComputing/ConditionalGaussianDistributions.pdf
        # slide 8 of 33
        assert given_rates.shape[0] < number_of_months_total,\
            "No variables left to sample."
        if self.SIGMA is None:
            self.populate_covariance(size=number_of_months_total)
        elif number_of_months_total != self.SIGMA.shape[0]:
            self.populate_covariance(size=number_of_months_total)
        if given_timepoints is None:
            given_timepoints = np.arange(given_rates.shape[0])
        elif isinstance(given_timepoints, int):
            given_timepoints = np.arange(given_timepoints)
        assert given_timepoints.shape[0] == given_rates.shape[0],\
            "There must be as many given timepoints as given values."
        assert np.unique(given_timepoints).shape[0] == given_timepoints.shape[0],\
            "All values must have a unique timepoint."
        
        given_timepoints_mask = np.zeros(number_of_months_total).astype(bool)
        given_timepoints_mask[given_timepoints] = True
        result_vector = np.zeros(number_of_months_total)
        LAMBDA_aa_inv = np.linalg.inv(
            self.LAMBDA[~given_timepoints_mask,:][:,~given_timepoints_mask])
        LAMBDA_ab = self.LAMBDA[~given_timepoints_mask,:][:,given_timepoints_mask]
        mu_ab = 0. - LAMBDA_aa_inv @ LAMBDA_ab @ (given_rates - 0.)
        result_vector[given_timepoints_mask] = given_rates
        Cholesky_L_LAMBDA_aa_inv = np.linalg.cholesky(LAMBDA_aa_inv)
        sample = np.random.randn(
            number_of_months_total-given_timepoints.shape[0],
            num_samples)
        sample = Cholesky_L_LAMBDA_aa_inv @ sample
        sample = sample + mu_ab[:,np.newaxis].repeat(num_samples,1)
        result_vector = result_vector[...,np.newaxis].repeat(num_samples,1)
        result_vector[~given_timepoints_mask,:] = sample
        return result_vector

    def refit_kernel(self, suppress=False):
        weights = np.sqrt(self.get_count_per_autocorrelation())
        def func_to_min(tau):
            x = np.arange(self.autocorrelation_maxlag)
            y = self.kernel_func(x, tau)
            return weights @ (self.autocorrelations - y)**2
        x0 = 0.01
        if self.kernel_type == 'mixture':
            x0 = np.array([x0,x0,0.5])
        self.tau = fmin(func=func_to_min, x0=x0, disp=not suppress).ravel()

# TODO: this does weird stuff. Why?
class IBORGaussianProccessML(IBORGaussianProccess):
    log_noise = None
    def __init__(
        self,
        historical_ibor,  # give in fractions not percentages
        autocorrelation_maxlag=100,
        kernel_type='laplace',
        fit=True,
        min_noise=0.01
        ):
        import warnings
        warnings.warn(
            "\nThere may be a bug in IBORGaussianProccessML."
            +"It yields strange fits, far noisier than the input data.")

        self.min_noise = min_noise
        super().__init__(
            historical_ibor=historical_ibor,
            autocorrelation_maxlag=autocorrelation_maxlag,
            kernel_type=kernel_type,
            fit=fit)

    def compute_log_marginal_likelihood(self):
        if self.SIGMA.shape[0] != self.historical_ibor.shape[0]:
            self.populate_covariance(size=self.historical_ibor.shape[0])
        lml = -0.5*self.historical_ibor @ self.LAMBDA @ self.historical_ibor\
                - 0.5*np.log(np.linalg.det(self.SIGMA)) \
                - (self.historical_ibor.shape[0]/2)*np.log(2*np.pi)
        return lml

    def populate_covariance(self, size=100):
        self.LAMBDA = None  # reset precision matrix
        self.SIGMA = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                self.SIGMA[i,j] = self.kernel(i,j)*self.standard_deviation**2
        # check positive definite
        EIGS, EIGVECS = np.linalg.eigh(self.SIGMA)
        if EIGS.min() < 1e-12:
            EIGS[EIGS < 1e-12] = 1e-12
            self.SIGMA = (EIGVECS @ np.diag(EIGS) @ EIGVECS.T)
        for i in range(self.SIGMA.shape[0]):
            self.SIGMA[i,i] += np.exp(self.log_noise) + self.min_noise
        self.LAMBDA = np.linalg.inv(self.SIGMA)
        return None

    def refit_kernel(self, suppress=False):
        """Uses the log-marginal-likelihood to fit the kernel."""
        def func_to_min(x):
            self.tau = x[:-1]
            self.log_noise = np.exp(x[-1])
            self.populate_covariance(len(self.historical_ibor))
            return -self.compute_log_marginal_likelihood()
        # fit tau and log_noise together
        x0 = np.array([0.01, 0.1])
        if self.kernel_type == 'mixture':
            x0 = np.array([0.01,0.01,0.5,0.1])
        xT = fmin(func=func_to_min, x0=x0, disp=not suppress).ravel()
        self.tau, self.log_noise = xT[:-1], xT[-1]

