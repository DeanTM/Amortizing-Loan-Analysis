from typing import Sequence
import numpy as np
from scipy.stats.kde import gaussian_kde



class Summary:
    _kde_estimate = None
    _cdf = None
    _percentages = None

    def __init__(
        self, values:Sequence,
        name=None,
        ):
        self.values = np.array(values)
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
