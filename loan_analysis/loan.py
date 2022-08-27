import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin
from scipy.stats.kde import gaussian_kde

from abc import ABC, abstractmethod

class AmortizingLoan:
    """A class that keeps track of the repayment of loans."""
    def __init__(self, initial_principal, full_term):
        self.initial_principal = initial_principal
        self.full_term = full_term
        self.reset()

    def reset(self):
        self.remaining_principals = [self.initial_principal]
        self.interests_paid = []
        self.principal_reductions = []
        self.payments_made = []
        self.remaining_term = self.full_term
    
    def split_repayment(self, payment, interest_rate_annual):
        remaining_principal = self.remaining_principals[-1]
        interest = remaining_principal * interest_rate_annual/12
        principal_reduction = payment - interest
        remaining_principal_new = remaining_principal - principal_reduction
        return interest, principal_reduction, remaining_principal_new

    def payment_update(self, payment, interest_rate_annual):
        interest, principal_reduction,\
        remaining_principal_new = self.split_repayment(
            payment, interest_rate_annual)
        self.interests_paid.append(interest)
        self.principal_reductions.append(principal_reduction)
        self.remaining_principals.append(remaining_principal_new)
        self.payments_made.append(payment)
        self.remaining_term -= 1
        return None

    def payment_update_collection(self, payments, interest_rates):
        for payment,interest_rate in zip(payments, interest_rates):
            self.payment_update(
                payment=payment,
                interest_rate_annual=interest_rate)
        return None

    def pay_interest_rates(self, interest_rates):
        for ir in interest_rates:
            payment = self.get_amortized_payment_amount(
                interest_rate_annual=ir)
            self.payment_update(
                payment=payment,
                interest_rate_annual=ir)
        return None


    def get_amortized_payment_amount(
        self,
        interest_rate_annual,
        ):
        # from here:
        # https://www.educba.com/amortized-loan-formula/
        r = interest_rate_annual
        n = 12
        num_payments_left = self.remaining_term
        t_times_n = num_payments_left
        P = self.remaining_principals[-1]
        return P * (r/n) * (1+r/n)**t_times_n /((1+r/n)**t_times_n -1)

    def get_data(self):
        data_dict = dict(
            payment_number=np.arange(len(self.payments_made)),
            principal_before=self.remaining_principals[:-1],
            principal_after=self.remaining_principals[1:],
            interest_paid=self.interests_paid,
            principal_paid=self.principal_reductions,
            total_paid=self.payments_made)
        return pd.DataFrame.from_dict(data_dict).set_index('payment_number')


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

def get_loan_rates(
    interest_rate_curve,
    spread=0.0,
    renew_every=1
    ):
    return np.repeat(
        interest_rate_curve[::renew_every],
        renew_every, axis=0) + spread


class IBORProcess(ABC):
    """Abstract base class for simulators of interbank offered rates."""
    @abstractmethod
    def sample_ior_curve(
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

    def __init__(
        self,
        historical_ibor,  # give in fractions not percentages
        autocorrelation_maxlag=100,
        kernel_type='laplace'
        ):
        self.historical_ibor = historical_ibor
        self.autocorrelation_maxlag = autocorrelation_maxlag
        self.kernel_type = kernel_type
        self.autocorrelations = get_autocorrelation(
            trace = self.historical_ibor,
            maxlag = self.autocorrelation_maxlag)
        self.standard_deviation = np.std(self.historical_ibor)
        self.refit_kernel(suppress=True)  #TODO: SUPPRESS TEXT
    
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
        else:
            raise NotImplementedError(f"Kernel {self.kernel} not implemented.")
    
    def kernel(self, i,j):
        """Computes inferred Covariance"""
        assert self.tau is not None, "Must fit first."
        return self.kernel_func(np.abs(i-j))*self.standard_deviation**2

    def populate_covariance(self, size=100):
        self.SIGMA = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                self.SIGMA[i,j] = self.kernel(i,j)
        # check positive definite
        EIGS, EIGVECS = np.linalg.eigh(self.SIGMA)
        if EIGS.min() < 1e-12:
            EIGS[EIGS < 1e-12] = 1e-12
            self.SIGMA = (EIGVECS @ np.diag(EIGS) @ EIGVECS.T)
        return self.SIGMA

    def sample_ior_curve(
        self,
        number_of_months,
        num_samples=1,
        initial_value=None
        ):
        refit_all = False
        if self.SIGMA is None:
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

    def refit_kernel(self, suppress=False):
        weights = np.sqrt(self.get_count_per_autocorrelation())
        def func_to_min(tau):
            x = np.arange(self.autocorrelation_maxlag)
            y = self.kernel_func(x, tau)
            return weights @ (self.autocorrelations - y)**2
        self.tau = fmin(func=func_to_min, x0=0.01, disp=not suppress)[0]

        

class Summary:
    _kde_estimate = None
    _cdf = None
    _percentages = None

    def __init__(self, values, name=None):
        self.values = values
        self.refit()
        self.name = name
        
    def refit(self):
        self._fit_kde()
        self._fit_cdf()

    def _fit_kde(self):
        self._kde_estimate = gaussian_kde(self.values)
        return None

    def get_kde(self):
        if self._kde_estimate is None:
            self._fit_kde()
        return self._kde_estimate
    
    def estimate_density(self, x):
        return self._kde_estimate(x)

    def _fit_cdf(
        self,
        percentages=np.linspace(0,100,1000)
        ):
        self._percentages = percentages
        self._cdf = np.percentile(self.values, self._percentages)
        return None

    def get_cdf(self):
        if self._cdf is None:
            self._fit_cdf()
        return self._cdf, self._percentages

    def get_var(self, percentage):
        if self._cdf is None:
            self._fit_cdf()
        var_idx = (self._percentages <= 100.-percentage).sum()
        return self._cdf[var_idx-1]

    




if __name__ == '__main__':
    ## example code
    principal = 250000.
    total_payemts = 480

    loan_35anos_variable = AmortizingLoan(
        initial_principal=principal,
        full_term=total_payemts)
    payment = 717.10
    interest_rate = 0.01566
    for i in range(total_payemts):
        loan_35anos_variable.payment_update(
            payment=payment,
            interest_rate_annual=interest_rate)



    fig = plt.figure(figsize=(8,3))
    plt.plot(
        np.cumsum(loan_35anos_variable.interests_paid),
        label='cumulative interest paid',
        color="C0")
    plt.plot(
        np.cumsum(loan_35anos_variable.principal_reductions),
        label='cumulative principal paid',
        color="C0", ls="--")
    plt.axhline(principal, ls=":", color='black')
    plt.xlabel("Month")
    plt.ylabel("Cumulative Payment (euros)")
    plt.legend()
    plt.show()
