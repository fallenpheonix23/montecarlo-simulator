"""Option payoff functions.

Each function takes a `paths` array and the relevant strike/barrier, and
returns a 1-D numpy array of the (undiscounted) payoff for every path.

Convention: `paths` has shape (n_paths, n_steps + 1), where column 0 is the
starting price S0 and the last column is the terminal price S_T.
"""

import numpy as np


def european_call(paths, K):
    """max(S_T - K, 0) -- depends only on the final price."""
    S_T = paths[:, -1]                      # terminal price of each path
    return np.maximum(S_T - K, 0.0)


def european_put(paths, K):
    """max(K - S_T, 0)."""
    S_T = paths[:, -1]
    return np.maximum(K - S_T, 0.0)


def asian_call(paths, K):
    """max(mean(S) - K, 0) -- mean is taken over the whole path."""
    S_avg = paths.mean(axis=1)              # average price along each path
    return np.maximum(S_avg - K, 0.0)


def asian_put(paths, K):
    """max(K - mean(S), 0)."""
    S_avg = paths.mean(axis=1)
    return np.maximum(K - S_avg, 0.0)


def barrier_knockout_call(paths, K, barrier):
    """European call that pays 0 if the price ever touches/crosses the barrier.

    This is a "down-and-out" call: if the price at ANY point in the path is at
    or below `barrier`, the option is knocked out and pays nothing.
    """
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0)       # start from the plain call payoff
    # A path is knocked out if it touches or drops below the barrier anywhere.
    knocked_out = np.any(paths <= barrier, axis=1)
    payoff[knocked_out] = 0.0               # zero out the knocked-out paths
    return payoff
