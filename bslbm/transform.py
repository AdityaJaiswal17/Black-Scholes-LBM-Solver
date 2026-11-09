import numpy as np
from math import sqrt, exp
from scipy.stats import norm

K, r, sigma, T = 100.0, 0.05, 0.20, 1.0

# x = ln(S/K), tau = sigma^2 (T - t) / 2, V = K exp(alpha x + beta tau) u
k = 2 * r / sigma**2
alpha = -(k - 1) / 2
beta = -(k + 1) ** 2 / 4
TAU_MAX = 0.5 * sigma**2 * T
XMIN, XMAX = -2.0, 2.0          # S from about 13.5 to 739

S_EVAL = np.linspace(50, 150, 201)


def bs_call(S, t=T):

    """ Closed-form Black-Scholes price of a European call.

    Parameters
    -----------
    S : float or array, spot price.
    t : float, time to expiry in years.

    Returns
    --------
    float or array, the call price.
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)


def payoff(x):

    """ The payoff max(S - K, 0) in heat-equation variables (tau = 0). """

    return np.maximum(np.exp((k + 1) * x / 2) - np.exp((k - 1) * x / 2), 0.0)


def far_field(x, tau):

    """ Deep in the money a call is worth S - K exp(-r (T - t)).

    This is that value in heat-equation variables, used as the right
    boundary. The discount on K matters: without it the boundary is wrong
    by K (1 - exp(-r (T - t))).
    """

    return (np.exp((k + 1) * x / 2 + (k + 1) ** 2 * tau / 4)
            - np.exp((k - 1) * x / 2 + (k - 1) ** 2 * tau / 4))


def to_price(x, u, tau=TAU_MAX):

    """ Converts a heat-equation solution back to spot and option price.

    Returns
    --------
    S, V : arrays, spot prices and call prices on the grid.
    """

    return K * np.exp(x), K * np.exp(alpha * x + beta * tau) * u


def rmse(x, u):

    """ RMSE against the closed form for S between 50 and 150. """

    S, V = to_price(x, u)
    error = np.interp(S_EVAL, S, V) - bs_call(S_EVAL)
    return float(np.sqrt(np.mean(error**2)))
