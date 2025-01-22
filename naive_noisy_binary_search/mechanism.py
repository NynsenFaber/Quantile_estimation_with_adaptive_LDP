import numpy as np
from collections import defaultdict
from .utils import vector_RR


def unbiased_probability(eps: float) -> callable:
    """
    Return an unbiased estimator for the head coin probability after randomized response
    :param eps: privacy budget

    :return: unbiased estimator for the sensitive head coin probability
    """
    # for unbiased estimation of probabilities
    factor_1 = (np.exp(eps) + 1) / (np.exp(eps) - 1)
    factor_2 = 1 / (np.exp(eps) + 1)
    return lambda x: factor_1 * (x - factor_2)


def naive_noisy_binary_search(data: np.array,
                              coins: np.array,
                              eps: float,
                              target: float,
                              replacement: bool = False,
                              test: bool = False) -> int:
    """
    Naive Noisy Binary Search algorithm with randomized response. It runs noisy binary search using the entire dataset.
    It returns the coin with empirical head probability closest to the target value.

    Parameters:
    ----------
    :param data: array of values
    :param coins: set of coins [1,2,3,...,B].
    :param eps: privacy parameter
    :param target: target or quantile in (0,1) value
    :param replacement: with (True) / without (False) replacement
    :param test: if True, it runs the test

    Output:
    ----------
    :return: a coin (int)
    """
    assert eps > 0, "Privacy parameter must be positive"
    assert 0 < target < 1, "Target must be between 0 and 1"
    assert type(coins) == np.ndarray, "Coins must be a numpy array"
    assert type(data) == np.ndarray, "Data must be a numpy array"

    data = np.array(data)
    M = len(data)  # number of steps

    coins.sort()  # make sure the coins are sorted
    B = len(coins)

    L = 0  # left index
    R = B - 1  # right index

    # split the number of iteration in log(B) steps
    max_steps = int(np.ceil(np.log2(B)))  # max number of steps for the noisy binary search
    M_flip = int(np.floor(M / max_steps))  # number of flips per step

    remaining_flips = int(M - max_steps * M_flip)  # remaining number of flips
    flag_redistribute = bool(remaining_flips > 0)  # if it is necessary to redistribute some flips

    # sample users in batch
    sample_indices = np.random.choice(len(data), M, replace=replacement)  # random permutation of users
    samples = None
    samples_list = None
    if flag_redistribute:
        # we add one additional flip at the start of the search
        sample_indices_1 = sample_indices[:remaining_flips * (M_flip + 1)]
        samples_1 = data[sample_indices_1].reshape(remaining_flips, M_flip + 1)
        sample_indices_2 = sample_indices[remaining_flips * (M_flip + 1):]
        samples_2 = data[sample_indices_2].reshape(max_steps - remaining_flips, M_flip)
        samples_list: list[np.array] = [samples_1] + [samples_2]
    else:
        samples: np.array = data[sample_indices].reshape(max_steps, M_flip)

    if test:
        # assert that the sizes are correct
        if samples is not None:
            assert samples.size == M, "The number of flips is not equal to M, samples"
        if samples_list is not None:
            sum_1 = np.sum(samples_list[0].size)
            sum_2 = np.sum(samples_list[1].size)
            assert sum_1 + sum_2 == M, "The number of flips is not equal to M, samples_list"

    # prepare the random variable for the DP coin
    eps = np.clip(eps, 0.0001, 100)  # for stability
    prob = np.exp(eps) / (1 + np.exp(eps))
    if flag_redistribute:
        random_coin_1 = np.random.binomial(1, prob, samples_list[0].shape)
        random_coin_2 = np.random.binomial(1, prob, samples_list[1].shape)
        random_coins = [random_coin_1] + [random_coin_2]
    else:
        random_coins = np.random.binomial(1, prob, samples.shape)

    # instatiate the unbiased estimator
    unbiased_estimator: callable = unbiased_probability(eps)

    # start with the middle coin
    index_coin = (L + R) // 2
    coin = coins[index_coin]

    count = 0
    if flag_redistribute:
        sample = samples_list[0]
        random_coin = random_coins[0]
    else:
        sample = samples
        random_coin = random_coins

    # when stop is reached, we need to use the second batch of samples with -1 samples
    # otherwise, it is necessary to stop the search
    stop = sample.shape[0]

    n_sampled = 0  # to check to sample the right number of elements

    while L <= R:  # start the noisy binary search

        # flip the coin as many times as sample size
        y: np.array = sample[count] <= coin

        n_sampled += len(y)  # update the number of sampled elements

        # apply randomized response to the coin flip
        y: np.array = vector_RR(y, random_coin[count])

        # compute the empirical cdf of the coin
        number_of_flips = len(y)  # update the number of tosses
        number_of_heads = np.sum(y)  # update the number of heads

        # compute the unbiased probability of the coin
        prob_coin = unbiased_estimator(number_of_heads / number_of_flips)

        if prob_coin > target:
            R = index_coin - 1
        elif prob_coin <= target:
            L = index_coin + 1

        # update coin
        index_coin = (L + R) // 2
        coin = coins[index_coin]

        count += 1

        if count == stop:
            if flag_redistribute:
                count = 0
                sample = samples_list[1]  # get second batch of samples
                random_coin = random_coins[1]  # get second batch of random coins
                stop = sample.shape[0]  # update stops
                flag_redistribute = False
            else:
                break

    if test: assert n_sampled == M, "The number of sampled elements is not equal to M"

    # return the last coin
    return coin
