import numpy as np
from collections import defaultdict
from .utils import vector_RR


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
