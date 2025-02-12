import numpy as np
from numba import njit
from tqdm import tqdm


# Use numba to speed up the code


def bayes_learn(data: list,
                alpha_update: float,
                eps: float,
                M: int,
                intervals: np.array,
                target: float = 0.5,
                replacement: bool = False) -> tuple[np.array, list]:
    """
    A differential private implementation of BayesLearn for the median
    given a dataset D (np.array) of values in [B], an error alpha in (0, 0.25),
    and a number of steps (samples from D) M smaller than the length of the data,
    returns a list of visited intervals.

    Ref:  Gretta, Lucas, and Eric Price. "Sharp Noisy Binary Search with Monotonic Probabilities."
          arXiv preprint arXiv:2311.00840 (2023).

    :param data: dataset of values in [B]
    :param alpha_update: error used to update the weights
    :param eps: privacy parameter
    :param M: steps (samples)
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param target: target or quantile in (0,1) value
    :param replacement: with (True) / without (False) replacement

    :return: L, list (np.array) of visited intervals indices
    :return: D, the dataset after the sampling
    """
    assert 0 < alpha_update < 0.25, "alpha must be in (0, 0.25) for the median"
    assert 0 <= M < len(data), "M must be smaller than the length of the data"

    if not replacement:
        # generate a random permutation of the data indices
        data = np.random.permutation(data)
        data_to_use = data[:M]  # data to use for bayes learning
        data_not_used = data[M:]  # data not used
    else:
        indices = np.random.choice(len(data), M, replace=True)  # generate M random indices with replacement
        data_to_use = data[indices]  # data to use for bayes learning
        data_not_used = data  # data not used

    # generate bernoulli coins with probability np.exp(eps) / (1 + np.exp(eps))
    eps = np.clip(eps, 0.00001, 100)  # for stability
    random_coin = np.random.binomial(1, np.exp(eps) / (1 + np.exp(eps)), M)

    # compute

    # define uniform prior over the intervals
    w: np.array = np.ones(len(intervals)) / (len(intervals))
    # create array for the visited intervals
    L = np.zeros(M, dtype=int)

    if target == 0.5:
        update: callable = update_weights_median
    else:
        raise NotImplementedError("BayesLearn is only implemented for the median")

    for i, sample in enumerate(data_to_use):
        # get the median interval and closest coin of the weights
        interval_index, coin_index = get_interval_coin_from_quantile(w, target, i)
        j_i = intervals[interval_index]
        x_i = j_i[coin_index]

        # add the interval index to the list
        L[i] = interval_index

        # flip the coin
        y_i = flip_coin(coin=x_i, sample=sample)

        # apply randomized response to the coin flip
        y_i = RR(y_i, random_coin[i])

        # update the weights (this can be optimized by using a segment tree as proposed in the reference paper)
        w = update(w, alpha_update, y_i, interval_index)

    return L, data_not_used


@njit
def get_interval_coin_from_quantile(w: np.array, q: float, i: int) -> tuple[int, int]:
    """
    Get the interval from the quantile q of the weights w.
    :param w: weights
    :param q: quantile
    :param i: iteration

    :return: interval index, coin index
    """
    assert 0 <= q <= 1

    if i == 0:
        # get the middle interval
        return len(w) // 2, 0

    W = np.cumsum(w)  # get cumulative sum of the weights
    i = np.argmax(W >= q)  # Find the first index where W[i] >= q
    coin = round_interval_to_coin(w[i], W[i], q)
    return i, coin


@njit
def round_interval_to_coin(w: float, W: float, q: float) -> int:
    """
    Round interval i to a coin.
    :param w: weights of the intervals
    :param W: cumulative sum of the weights
    :param q: quantile
    :return: coin index {0,1}
    """

    flag = (q - W) / w - q
    if flag <= 0:
        # return left coin of the interval
        return 0
    else:
        # return right coin of the interval
        return 1


def flip_coin(coin: int, sample: int) -> int:
    return int(sample <= coin)


@njit
def RR(A: int, B: int) -> int:
    """
    Given the result of coin A and the result of DP coin due to randomized response B, we return the private coin
    Truth Table:
    A  B  result
    0  0  1
    0  1  0
    1  0  0
    1  1  1
    So if the DP coin is zero we flip coin A, otherwise we keep the result of coin A
    :param A:
    :param B:
    :return:
    """
    A = bool(A)
    B = bool(B)
    return int((A and B) or (not A and not B))


@njit
def update_weights_median(w: np.array, alpha: float, y: int, j: int) -> np.array:
    """
    Update the weights of the intervals. Specialized for the median.
    :param w: weights
    :param alpha: target error
    :param y: result of the comparison (0 or 1)
    :param j: interval chosen
    :return: updated weights

    Note that this function can be optimized in running time from O(B) to O(log B) by using a lazily initialized
    segment tree as proposed in the reference paper
    """
    # update left
    w[:j] *= (1 - 2 * alpha * (-1) ** y)
    # update right
    w[j + 1:] *= (1 + 2 * alpha * (-1) ** y)
    S = np.sum(w[:j]) + np.sum(w[j + 1:])
    # add the renormalization step
    w[j] = 1 - S
    return w



