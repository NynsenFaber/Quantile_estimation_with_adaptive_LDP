import numpy as np
from collections import defaultdict
from .utils import get_q, get_d00, get_d01, get_d10, get_d11


def median_bayes_learn(D: list,
                       alpha: float,
                       eps: float,
                       M: int,
                       intervals: np.array,
                       replacement: bool = False) -> tuple[np.array, dict, list]:
    """
    BayesLearn for the median given a dataset D (np.array) of values in [B], an error alpha in (0, 0.5),
    and a number of steps (samples from D) M smaller than the length of the data,
    returns a list of visited intervals and the empirical probability of the coin.
    probabilities
    :param D: dataset of values in [B]
    :param alpha: target error
    :param eps: privacy parameter
    :param M: steps (samples)
    :param intervals: list of intervals
    :param replacement: with (True) / without (False) replacement
    :return: L, list (np.array) of visited intervals
    :return: H, histogram of the probabilities sampled
    :return: D, the dataset after the sampling
    """
    assert 0 < alpha < 0.5
    assert 0 <= M <= len(D)
    D = list(D)

    # generate bernoulli coins with probability np.exp(eps) / (1 + np.exp(eps))
    eps = np.clip(eps, 0.00001, 100)  # for stability
    random_coin = np.random.binomial(1, np.exp(eps) / (1 + np.exp(eps)), M)

    # define uniform prior
    w: np.array = np.ones(len(intervals)) / (len(intervals))
    # create array for the visited intervals
    L = np.zeros(M)
    # create dict to count the coins visited, initialize with 0 toss and 0 heads
    coins_count = defaultdict(lambda: [0, 0])

    for i in range(M):
        # get the median interval and closest coin of the weights
        interval_index, coin_index = get_interval_coin_from_quantile(w, 0.5)
        j_i = intervals[interval_index]
        x_i = j_i[coin_index]

        # update coins_count
        coins_count[x_i][0] += 1

        # add the interval index to the list
        L[i] = interval_index

        # sample a random element from D
        sample = sample_element(D=D, replacement=replacement)

        # flip the coin
        y_i = flip_coin(coin=x_i, sample=sample)

        # apply randomized response to the coin flip
        y_i = RR(y_i, random_coin[i])

        # update the histogram by adding the result of the coin flip
        coins_count[x_i][1] += y_i

        # update the weights (this can be optimized by using a segment tree as proposed in the reference paper)
        w = update_weights_median(w, alpha, y_i, interval_index)

    return L.astype(int), coins_count, D


# to fix flip coin
def bayes_learn(D: np.array, tau: float, alpha: float, M: int, B: int, replacement: bool = False) -> np.array:
    """
    BayesLearn, given a dataset D of values in [B], a target probability tau, an error alpha,
    and a number of steps (samples from D) M, returns a list of visited intervals.
    :param D: dataset of values in [B]
    :param tau: target probability
    :param alpha: target error
    :param M: steps (samples)
    :param B: number of coins
    :param replacement: with (True) / without (False) replacement
    :return: L, list (np.array) of visited intervals
    """
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1
    assert 0 <= M <= len(D)

    # define uniform prior
    w: np.array = np.ones(B - 1) / (B - 1)  # len(D)-1 because we are considering intervals
    q = get_q(tau, alpha)
    # create array for the visited intervals
    L = np.zeros(M)
    # create dict to count the coins visited
    coins_count = {}
    for i in range(M):
        # get the median interval and closest coin of the weights
        j_i, x_i = get_interval_coin_from_quantile(w, q)
        # update coins_count
        if x_i in coins_count:
            coins_count[x_i][0] += 1
        else:
            coins_count[x_i] = [1, 0]
        # print(f"round {i}: interval {j_i}, coin {x_i}")
        L[i] = j_i
        # flip the coin
        y_i, D = flip_coin(D, x_i, replacement=replacement)
        # update the histogram
        coins_count[x_i][1] += y_i
        # update the weights
        w = update_weights(w, tau, alpha, y_i, j_i)
    # sort H and coins_count by keys
    coins_count = dict(sorted(coins_count.items()))

    return L.astype(int), coins_prob


def get_interval_coin_from_quantile(w: np.array, q: float) -> tuple[int, int]:
    """
    Get the interval from the quantile q of the weights w.
    :param w: weights
    :param q: quantile
    :return: interval, coin
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
    :return: coin
    """

    flag = (q - W) / w[i] - q
    if flag <= 0:
        return 0
    else:
        return 1


def flip_coin(coin: int, sample: int) -> int:
    return int(sample <= coin)


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


def update_weights(w: np.array, tau: float, alpha: float, y: int, j: int) -> np.array:
    """
    Update the weights of the intervals.
    :param w: weights
    :param tau: target probability
    :param alpha: target error
    :param y: result of the comparison (0 or 1)
    :param j: interval chosen
    :return: updated weights

    Note that this function can be optimized in running time from O(B) to O(log B) by using a lazily initialized
    segment tree as proposed in the reference paper
    """
    if y == 0:
        left_weight: callable = get_d00
        right_weight: callable = get_d01
    else:
        left_weight: callable = get_d10
        right_weight: callable = get_d11

    # update left
    weight = left_weight(tau, alpha)
    w[:j] *= weight
    # update right
    weight = right_weight(tau, alpha)
    w[j + 1:] *= weight
    # normalize
    S = np.sum(w[:j]) + np.sum(w[j + 1:])
    w[j] = 1 - S
    return w


def sample_element(D: list, replacement):
    sample_index = np.random.choice(len(D), 1)[0]
    sample = D[sample_index]
    if not replacement:
        del D[sample_index]
    return sample


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
