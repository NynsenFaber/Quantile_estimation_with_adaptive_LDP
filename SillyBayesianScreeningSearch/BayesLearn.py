import numpy as np
from utils import get_q


def bayes_learn(D: np.array, tau: float, alpha: float, M: int) -> np.array:
    """
    BayesLearn, given a dataset D of values in [B], a target probability tau, an error alpha,
    and a number of steps (samples from D) M, returns a list of visited intervals.
    :param D: dataset of values in [B]
    :param tau: target probability
    :param alpha: target error
    :param M: steps (samples)
    :return: L, list (np.array) of visited intervals
    """
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1
    assert 0 <= M <= len(D)

    # define uniform prior
    w: np.array = np.ones(len(D) - 1) / (len(D) - 1)  # len(D)-1 because we are considering intervals
    q = get_q(tau, alpha)
    L = np.zeros(M)
    for i in range(M):
        j_i = get_interval_from_quantile(w, q)
        x_i = round_interval_to_coin(j_i, w, q)


def get_interval_from_quantile(w: np.array, q: float) -> np.array:
    """
    Get the interval from the quantile q of the weights w.
    :param w: weights
    :param q: quantile
    :return: interval
    """
    assert 0 <= q <= 1

    W = np.cumsum(w)  # get cumulative sum
    for i in range(len(w)):
        # return minimum i such that W[i] >= q
        if W[i] >= q:
            return i
    return len(w) - 1


def round_interval_to_coin(i: int, w: np.array, q: float) -> int:
    """
    Round interval i to a coin.
    :param i: input interval
    :param w: weights of the intervals
    :param q: quantile
    :return: coin
    """
    assert 0 <= i < len(w)
    assert 0 <= q <= 1

    W = np.cumsum(w)  # get cumulative sum
    flag = (q - W[i]) / w[i] - q  # ?
    if flag <= 0:
        return i
    else:
        return i + 1
