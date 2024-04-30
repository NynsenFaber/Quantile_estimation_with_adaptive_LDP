import numpy as np
from .utils import get_q, get_d00, get_d01, get_d10, get_d11


def median_bayes_learn(D: np.array, alpha: float, M: int, replacement: bool = False) -> np.array:
    """
    BayesLearn for the median given a dataset D of values in [B], a target probability tau, an error alpha,
    and a number of steps (samples from D) M, returns a list of visited intervals.
    :param D: dataset of values in [B]
    :param alpha: target error
    :param M: steps (samples)
    :param replacement: with (True) / without (False) replacement
    :return: L, list (np.array) of visited intervals
    """
    assert 0 <= alpha <= 1
    assert 0.5 - alpha >= 0
    assert 0.5 + alpha <= 1
    assert 0 <= M <= len(D)

    # define uniform prior
    w: np.array = np.ones(len(D) - 1) / (len(D) - 1)  # len(D)-1 because we are considering intervals
    L = np.zeros(M)
    for i in range(M):
        # get the median interval and closest coin of the weights
        j_i, x_i = get_interval_coin_from_quantile(w, 0.5)
        print(f"round {i}: interval {j_i}, coin {x_i}")
        L[i] = j_i
        # flip the coin
        y_i, D = flip_coin(D, x_i, replacement=replacement)
        # update the weights
        w = update_weights_median(w, alpha, y_i, j_i)
    return L


def bayes_learn(D: np.array, tau: float, alpha: float, M: int, replacement: bool = False) -> np.array:
    """
    BayesLearn, given a dataset D of values in [B], a target probability tau, an error alpha,
    and a number of steps (samples from D) M, returns a list of visited intervals.
    :param D: dataset of values in [B]
    :param tau: target probability
    :param alpha: target error
    :param M: steps (samples)
    :param replacement: with (True) / without (False) replacement
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
        j_i, x_i = get_interval_coin_from_quantile(w, q)
        x_i = round_interval_to_coin(j_i, w, q)
        print(f"round {i}: interval {j_i}, coin {x_i}")
        L[i] = j_i
        # flip the coin
        y_i, D = flip_coin(D, x_i, replacement=replacement)
        # update the weights
        w = update_weights(w, alpha, y_i, j_i)
    return L


def get_interval_coin_from_quantile(w: np.array, q: float) -> tuple[int, int]:
    """
    Get the interval from the quantile q of the weights w.
    :param w: weights
    :param q: quantile
    :return: interval, coin
    """
    assert 0 <= q <= 1

    W = np.cumsum(w)  # get cumulative sum
    for i in range(len(w)):
        # return minimum i such that W[i] >= q
        if W[i] >= q:
            interval = i
            # compute the coin
            coin = round_interval_to_coin(i, w, 0.5)
            return i, coin
    raise ValueError(f"Quantile {q} not found in the weights w.")


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


def flip_coin(D: np.array, i: int, replacement: bool = False) -> tuple[int, np.array]:
    """
    Sample with / without replacement from the interval dataset (np.array) D a random user (a random value in [B]),
    compare the user with the coin (query: are you smaller than the coin?), and return the result.
    If without replacement, drop the user from the dataset.
    :param D: dataset of values in [B]
    :param i: coin
    :param replacement: with / without replacement
    :return: result of the comparison and the new dataset
    """
    assert 0 <= i < len(D)
    sample = np.random.choice(D, 1)
    if not replacement:
        # drop one element equal to sample from D
        index_to_drop = np.where(D == sample)[0][0]
        D = np.delete(D, index_to_drop)

    return int(sample <= i), D


def update_weights_median(w: np.array, alpha: float, y: int, j: int) -> np.array:
    """
    Update the weights of the intervals. Specialized for the median.
    :param w: weights
    :param alpha: target error
    :param y: result of the comparison (0 or 1)
    :param j: interval chosen
    :return: updated weights
    """
    sum_1 = 0
    sum_2 = 0
    for i in range(j):
        w[i] = w[i] * (1 - 2 * alpha * (-1) ** y)
        sum_1 += w[i]
    for i in range(j + 1, len(w)):
        w[i] = w[i] * (1 + 2 * alpha * (-1) ** y)
        sum_2 += w[i]
    # add the renormalization step
    w[j] = 1 - sum_1 - sum_2
    return w


def update_weights(w: np.array, tau: float, alpha: float, y: int, j: int) -> np.array:
    """
    Update the weights of the intervals.
    :param w: weights
    :param tau: target probability
    :param alpha: target error
    :param y: result of the comparison (0 or 1)
    :param j: interval chosen
    :return: updated weights
    """
    sum_1 = 0
    sum_2 = 0
    if y == 0:
        left_weight: callable = get_d00
        right_weight: callable = get_d01
    else:
        left_weight: callable = get_d10
        right_weight: callable = get_d11

    for i in range(j):
        w[i] = w[i] * left_weight(tau, alpha)
        sum_1 += w[i]

    for i in range(j + 1, len(w)):
        w[i] = w[i] * right_weight(tau, alpha)
        w[i] = w[i] * (1 + 2 * alpha * (-1) ** y)
        sum_2 += w[i]

    # add the renormalization step
    w[j] = 1 - sum_1 - sum_2
    return w
