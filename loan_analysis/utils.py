import matplotlib.pyplot as plt
import numpy as np

from .loan import AmortizingLoan
from .summary import Summary


def plot_mean_and_95_percentile(
    data,
    ax=None,
    colour=None,
    label=None,
    percentile=5
    ):
    if ax is None:
        ax = plt.gca()
    mean = np.mean(data, axis=1)
    percentile_low = np.percentile(data,percentile/2,axis=1)
    percentile_high = np.percentile(data,100-percentile/2,axis=1)
    ax.fill_between(
        np.arange(data.shape[0]),
        percentile_low,
        percentile_high,
        color=colour,
        alpha=0.5,
        label=label)
    ax.plot(
        np.arange(data.shape[0]),
        mean,
        color=colour)
    return ax



def plot_mean_pm_std(
    data,
    ax=None,
    colour=None,
    label=None
    ):
    if ax is None:
        ax = plt.gca()
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    ax.fill_between(
        np.arange(data.shape[0]),
        mean-std,
        mean+std,
        color=colour,
        alpha=0.5,
        label=label)
    ax.plot(
        np.arange(data.shape[0]),
        mean,
        color=colour)
    return ax


def plot_repayments(
    loan: AmortizingLoan,
    colour_interest=None,
    colour_principal=None,
    ax=None,
    show_legend=True
    ):
    if ax is None:
        ax = plt.gca()
    ax.plot(
        np.cumsum(loan.interests_paid),
        label='cumulative interest paid',
        color=colour_interest)
    ax.plot(
        np.cumsum(loan.principal_reductions),
        label='cumulative principal paid',
        color=colour_principal, ls="--")
    ax.axhline(loan.initial_principal, ls=":", color='black')
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative Payment (euros)")
    ax.grid(":")
    if show_legend:
        ax.legend(loc="upper left")
    return ax


def plot_summary(
    summary: Summary,
    plot_hist: bool=True,
    plot_cdf: bool=True,
    plot_kde: bool=True,
    var_percent: float=5.,
    ax=None
    ):
    if ax is None:
        ax = plt.gca()
    var = summary.get_var(var_percent)
    ax.axvline(
        var, ls=":", color="black",
        label=f"{var_percent}%-VaR: {var:.2f}")
    if plot_hist:
        bins = np.sqrt(len(summary.values)).astype(int)
        heights, edges = np.histogram(
            summary.values, bins=bins, density=True)
        mask = edges[:-1] <= var
        ax.bar(
            edges[:mask.sum()], heights[mask],
            width=np.diff(edges[:mask.sum()+1]),
            align="edge", color="maroon", alpha=0.8)
        ax.bar(
            edges[mask.sum():-1], heights[~mask],
            width=np.diff(edges[mask.sum():]),
            align="edge", color="navy", alpha=0.8)
    if plot_kde:
        kde = summary.get_kde()
        x = np.linspace(summary.values.min(), summary.values.max(), 1000)
        y = kde(x)
        mask = x <= var
        ax.plot(x[mask], y[mask], color='gray')
        ax.plot(x[~mask], y[~mask], color='black')
    if plot_cdf:
        cdf, percentages = summary.get_cdf()
        top_index = (percentages <= 100-var_percent).sum()
        ax2 = ax.twinx()
        ax2.plot(cdf[:top_index], percentages[:top_index], color='gray')
        ax2.plot(cdf[top_index-1:], percentages[top_index-1:], color='black')
    if summary.name:
        ax.set_title(summary.name)
    ax.legend(loc="upper left")
