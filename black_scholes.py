"""Analytical Black-Scholes pricer.

This module is used ONLY for validation -- it gives us the exact closed-form
price of a European option so we can confirm the Monte Carlo engine converges
to the right answer.
"""

import numpy as np
from scipy.stats import norm


def bs_price(S0, K, r, sigma, T, option_type):
    """Exact Black-Scholes price for a European call or put.

    Parameters
    ----------
    S0 : float        Current price of the underlying.
    K : float         Strike price.
    r : float         Risk-free interest rate (annualised, continuous).
    sigma : float     Volatility (annualised).
    T : float         Time to maturity in years.
    option_type : str 'call' or 'put'.

    Returns
    -------
    float : the analytical option price.
    """
    # d1 and d2 are the standard Black-Scholes terms.
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        # Call = S0 * N(d1) - K * e^{-rT} * N(d2)
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        # Put = K * e^{-rT} * N(-d2) - S0 * N(-d1)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
