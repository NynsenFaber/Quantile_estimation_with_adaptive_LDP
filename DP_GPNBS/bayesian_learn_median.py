import numpy as np


def bayes_learn(D: list,
                alpha: float,
                eps: float,
                M: int,
                intervals: np.array,
                replacement: bool = False) -> tuple[np.array, list]:
    """
    BayesLearn for the median given a dataset D (np.array) of values in [B], an error alpha in (0, 0.5),
    and a number of steps (samples from D) M smaller than the length of the data,
    returns a list of visited intervals and the empirical probability of the coin.
    probabilities
    :param D: dataset of values in [B]
    :param alpha: target error
    :param eps: privacy parameter
    :param M: steps (samples)
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param replacement: with (True) / without (False) replacement

    :return: L, list (np.array) of visited intervals indices
    :return: D, the dataset after the sampling
    """
    assert 0 < alpha < 0.5, "alpha must be in (0, 0.5)"
    assert 0 <= M <= len(D), "M must be smaller than the length of the data"
    assert type(D) == list, "D must be a list"

    # generate bernoulli coins with probability np.exp(eps) / (1 + np.exp(eps))
    eps = np.clip(eps, 0.00001, 100)  # for stability
    random_coin = np.random.binomial(1, np.exp(eps) / (1 + np.exp(eps)), M)

    # define uniform prior
    w: np.array = np.ones(len(intervals)) / (len(intervals))
    # create array for the visited intervals
    L = np.zeros(M)

    for i in range(M):
        # get the median interval and closest coin of the weights
        interval_index, coin_index = get_interval_coin_from_quantile(w, 0.5)
        j_i = intervals[interval_index]
        x_i = j_i[coin_index]

        # add the interval index to the list
        L[i] = interval_index

        # sample a random element from D
        sample = sample_element(D=D, replacement=replacement)

        # flip the coin
        y_i = flip_coin(coin=x_i, sample=sample)

        # apply randomized response to the coin flip
        y_i = RR(y_i, random_coin[i])

        # update the weights (this can be optimized by using a segment tree as proposed in the reference paper)
        w = update_weights(w, alpha, y_i, interval_index)

    return L.astype(int), D


def get_interval_coin_from_quantile(w: np.array, q: float) -> tuple[int, int]:
    """
    Get the interval from the quantile q of the weights w.
    :param w: weights
    :param q: quantile
    :return: interval index, coin index
    """
    assert 0 <= q <= 1

    W = np.cumsum(w)  # get cumulative sum of the weights
    i = np.argmax(W >= q)  # Find the first index where W[i] >= q
    coin = round_interval_to_coin(i, w, W[i], 0.5)
    return i, coin


def round_interval_to_coin(i: int, w: np.array, W, q: float) -> int:
    """
    Round interval i to a coin.
    :param i: input interval
    :param w: weights of the intervals
    :param W: cumulative sum of the weights
    :param q: quantile
    :return: coin index {0,1}
    """

    flag = (q - W) / w[i] - q
    if flag <= 0:
        return 0
    else:
        return 1


def sample_element(D: list, replacement):
    """
    Sample a random element from the dataset D
    :param D: dataset
    :param replacement: with (True) / without (False) replacement
    :return: an element from the dataset
    """
    sample_index = np.random.choice(len(D), 1)[0]
    sample = D[sample_index]
    if not replacement:
        del D[sample_index]
    return sample


def flip_coin(coin: int, sample: int) -> int:
    return int(sample <= coin)


def RR(A: int, B: int) -> int:
    """
    Given the result of coin A and the reuslt of DP coin due to randomized response B, we return the private coin
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


def update_weights(w: np.array, alpha: float, y: int, j: int) -> np.array:
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
