import numpy as np
from collections import defaultdict
from .utils import check_coins, vector_RR


def advanced_noisy_binary_search(data: np.array,
                                 intervals: np.array,
                                 M: int,
                                 alpha: float,
                                 eps: float,
                                 target: float,
                                 replacement: bool = False) -> int:
    """
    Naive Noisy Binary Search algorithm. It returns a coin that is an \alpha/2 good estimate of the target.
    Each coin during the binary search is sampled O(M/log(B)) times, allowing to perform a noisy binary search with O(log(B))
    steps. If a coin is \alpha/2 good, it is kept and resampled, otherwise, the search continues.
    At the end of the search, the algorithm returns a coin that is \alpha/2 good.

    :param data: array of values in [B]
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param M: number of iteration
    :param alpha: error
    :param eps: privacy parameter
    :param target: target or quantile in (0,1) value
    :param replacement: with (True) / without (False) replacement
    :return: a coin (int)
    """
    assert 0 < target < 1, "Target must be between 0 and 1"

    data = np.array(data)

    # for unbiased estimation of probabilities
    factor_1 = (np.exp(eps) + 1) / (np.exp(eps) - 1)
    factor_2 = 1 / (np.exp(eps) + 1)
    unbiased_prob: callable(float) = lambda x: factor_1 * (x - factor_2)

    # split the number of iteration in log(B) steps
    coins = list(set(intervals.flatten()))
    coins.sort()
    B = len(coins)
    L = coins[0]
    R = coins[-1]

    # for noisy binary search we get a (1/2)\alpha good coin for the estimate,
    # getting then an \alpha good coin for the final result.
    alpha_search = alpha * (1 / 2)

    M_search = int(M / np.log2(B))  # number of coin flips per step
    max_steps = M // M_search  # max number of steps for the noisy binary search

    # start with the middle coin (it has to be an array)
    coin = [coins[B // 2]]

    # prepare the random variable for the DP coin
    eps = np.clip(eps, 0.0001, 100)  # for stability
    random_coin = np.random.binomial(1, np.exp(eps) / (1 + np.exp(eps)), (max_steps, M_search))
    last_random_coin = np.random.binomial(1, np.exp(eps) / (1 + np.exp(eps)), M - max_steps * M_search)

    # sample users in batch
    sample_indices = np.random.choice(len(data), max_steps * M_search, replace=replacement)
    samples = data[sample_indices].reshape(max_steps, M_search)
    data = np.delete(data, sample_indices)
    last_sample = np.random.choice(data, M - max_steps * M_search, replace=replacement)

    # initialize the histogram (key: coin, value: [toss, heads])
    H = defaultdict(lambda: [0, 0])
    coins_prob = {}

    for i in range(max_steps):
        # flip the coin
        y = samples[i] <= coin

        # apply randomized response to the coin flip
        y = vector_RR(y, random_coin[i])

        # compute the empirical cdf of the coin
        H[coin[0]][0] += M_search
        H[coin[0]][1] += np.sum(y)

        # compute the unbiased probability of the coin
        coins_prob[coin[0]] = unbiased_prob(H[coin[0]][1] / H[coin[0]][0])

        if coins_prob[coin[0]] > target + alpha_search:
            R = coin[0]
            new_coin = (L + R) // 2
            # get the closest element in coins
            coin = [min(coins, key=lambda x: abs(x - new_coin))]
        elif coins_prob[coin[0]] < target - alpha_search:
            L = coin[0]
            new_coin = (L + R) // 2
            # get the closest element in coins
            coin = [min(coins, key=lambda x: abs(x - new_coin))]
        else:
            # resample the coin
            continue

        if L >= R:
            # resample the coin
            continue

    # sample the last users
    y = last_sample <= coin[0]
    y = vector_RR(y, last_random_coin)

    if M - max_steps * M_search > 0:
        H[coin[0]][0] += M - max_steps * M_search
        H[coin[0]][1] += np.sum(y)
        coins_prob[coin[0]] = unbiased_prob(H[coin[0]][1] / H[coin[0]][0])

    visited_coin = list(coins_prob.keys())
    random_good_coin = check_coins(alpha=alpha, coins=visited_coin, coins_prob=coins_prob, target=target)
    return random_good_coin
