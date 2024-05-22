import numpy as np
from collections import defaultdict


def check_coins(alpha: float,
                alpha_bayes: float,
                intervals: np.array,
                coins_prob: dict[int, float],
                target: float = 0.5) -> int:
    """
    Check if a coin is good or not, return a random good coin
    :param alpha: error to seek
    :param alpha_bayes: error for the Bayesian learning
    :param intervals: list of the intervals
    :param coins_prob: empirical cdf of the coins
    :param target: target or quantile in (0,1) value

    :return: a random good coin
    """
    good_coins = []
    left_bound = target - alpha + (alpha - alpha_bayes) / 2
    right_bound = target + alpha - (alpha - alpha_bayes) / 2
    for coin_left, coin_right in intervals:
        if coins_prob[coin_right] >= right_bound or coins_prob[coin_right] <= left_bound:
            continue
        else:
            good_coins.append(coin_left)
    if len(good_coins) > 0:
        random_good_coin = np.random.choice(good_coins, 1)[0]
        return random_good_coin
    else:
        # return the coin with the probability closest to the target
        closest_coin = min(coins_prob, key=lambda x: abs(coins_prob[x] - target))
        return closest_coin


def vector_RR(A: np.array, B: np.array) -> np.array:
    """
    Given the result of coin A and the result of DP coin due to randomized response B, we return the private coin
    Truth Table:
    A  B  result
    0  0  1
    0  1  0
    1  0  0
    1  1  1
    So if the DP coin is zero we flip coin A, otherwise we keep the result of coin A
    :param A: array-like, input coin A
    :param B: array-like, input DP coin B
    :return: array-like, private coin
    """
    A = np.asarray(A, dtype=bool)
    B = np.asarray(B, dtype=bool)
    return ((A & B) | (~A & ~B)).astype(int)


def get_intervals(bins: np.array) -> np.array:
    """
    Return intervals from bins
    """
    intervals = np.array([bins[:-1], bins[1:]]).T
    return intervals


def get_th_alpha(B: int, N: int, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (np.sqrt(N))


def naive_noisy_binary_search(data: np.array,
                              intervals: np.array,
                              M: int,
                              eps: float,
                              target: float,
                              replacement: bool = False) -> int:
    """
    Naive Noisy Binary Search algorithm. It runs noisy binary search with a fixed number of steps M.

    :param data: array of values in [B]
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param M: number of iteration
    :param eps: privacy parameter
    :param target: target or quantile in (0,1) value
    :param replacement: with (True) / without (False) replacement
    :return: a coin (int)
    """
    assert 0 < target < 1, "Target must be between 0 and 1"
    assert len(data) == M, "The number of elements in the dataset must be equal to M"

    data = np.array(data)

    # for unbiased estimation of probabilities
    factor_1 = (np.exp(eps) + 1) / (np.exp(eps) - 1)
    factor_2 = 1 / (np.exp(eps) + 1)
    unbiased_prob: callable(float) = lambda x: factor_1 * (x - factor_2)

    # split the number of iteration in log(B) steps
    coins = list(set(intervals.flatten()))
    coins.sort()
    B = len(coins)

    L = 0
    R = B - 1

    max_steps = int(np.floor(np.log2(B)))  # max number of steps for the noisy binary search
    M_search = int(np.floor(M / max_steps))
    remaining_flips = int(M - max_steps * M_search)  # remaining flips
    flag_redistribute = bool(remaining_flips > 0)

    # sample users in batch
    sample_indices = np.random.choice(len(data), M, replace=replacement)
    if flag_redistribute:
        sample_indices_1 = sample_indices[:remaining_flips * (M_search + 1)]
        samples_1 = data[sample_indices_1].reshape(remaining_flips, M_search + 1)
        sample_indices_2 = sample_indices[remaining_flips * (M_search + 1):]
        samples_2 = data[sample_indices_2].reshape(max_steps - remaining_flips, M_search)
        samples = [samples_1] + [samples_2]
    else:
        samples = data[sample_indices].reshape(max_steps, M_search)

    # prepare the random variable for the DP coin
    eps = np.clip(eps, 0.0001, 100)  # for stability
    prob = np.exp(eps) / (1 + np.exp(eps))
    if flag_redistribute:
        random_coin_1 = np.random.binomial(1, prob, samples[0].shape)
        random_coin_2 = np.random.binomial(1, prob, samples[1].shape)
        random_coins = [random_coin_1] + [random_coin_2]
    else:
        random_coins = np.random.binomial(1, prob, samples.shape)

    # initialize the histogram (key: coin, value: [toss, heads])
    H = defaultdict(lambda: [0, 0])
    coins_prob = {}

    # start with the middle coin (it has to be an array)
    coin = [coins[(L + R) // 2]]
    index_coin = (L + R) // 2

    count = 0
    if flag_redistribute:
        sample = samples[0]
        random_coin = random_coins[0]
    else:
        sample = samples
        random_coin = random_coins

    stop = sample.shape[0]
    n_sampled = 0  # to check to sample the right number of elements
    while L <= R:

        # flip the coin
        y = sample[count] <= coin

        n_sampled += len(y)

        # apply randomized response to the coin flip
        y = vector_RR(y, random_coin[count])

        # compute the empirical cdf of the coin
        H[coin[0]][0] += M_search
        H[coin[0]][1] += np.sum(y)

        # compute the unbiased probability of the coin
        coins_prob[coin[0]] = unbiased_prob(H[coin[0]][1] / H[coin[0]][0])

        if coins_prob[coin[0]] > target:
            R = index_coin - 1
        elif coins_prob[coin[0]] < target:
            L = index_coin + 1

        index_coin = (L + R) // 2
        coin = [coins[index_coin]]

        count += 1

        if count == stop:
            if flag_redistribute:
                count = 0
                sample = samples[1]
                random_coin = random_coins[1]
                stop = sample.shape[0]
                flag_redistribute = False
            else:
                break

    assert n_sampled == M, "The number of sampled elements is not equal to M"

    # return the last coin
    return coin[0]
