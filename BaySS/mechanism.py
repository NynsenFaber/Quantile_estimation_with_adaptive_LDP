import numpy as np
from .bayesian_learning import bayes_learn
from collections import defaultdict
from .utils import check_coins, vector_RR, naive_noisy_binary_search


def bayss_dp(data: list,
             alpha: float,
             eps: float,
             M: int,
             intervals: np.array,
             target: float = 0.5,
             replacement: bool = False,
             naive_NBS: bool = True) -> int:
    """
    Algorithm developed by Gretta-Price. It uses Bayesian learning to reduce the number of intervals. This version works
    with differential privacy and sampling without replacement.

    Ref:  Gretta, Lucas, and Eric Price. "Sharp Noisy Binary Search with Monotonic Probabilities."
          arXiv preprint arXiv:2311.00840 (2023).

    :param data: dataset of values in [B]
    :param alpha: target error
    :param eps: privacy parameter
    :param M: steps (samples)
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param target: target or quantile in (0,1) value
    :param replacement: with (True) / without (False) replacement
    :param naive_NBS: use the naive noisy binary search (True) for the final step, otherwise use flips all the coins
        in the final step and return the closest to the target.

    :return: the median
    """

    data = list(data)

    coins = list(set(intervals.flatten()))
    coins.sort()
    B = len(coins)

    # get the number of samples at for each step of the Bayesian learning
    # to have M_bayes_1/M_bayes_2 = log(B)/log(log(B)), M_bayes_1/M_sampling = log(B)
    # and M_bayes_2/M_sampling = log(log(B))
    den = np.log(B) + np.log(np.log(B)) + 1
    M_bayes_1 = int(M * np.log(B) / den)
    M_bayes_2 = int(M * np.log(np.log(B)) / den)

    flag = False  # to keep track if a second Bayesian learning is needed

    # define the alpha for the Bayesian learning and the gamma for the reduction
    alpha_bayes = alpha * (2 / 3)
    gamma = 1 / (np.log(B) ** 2)  # as proposed in the reference paper

    # get the intervals from the Bayesian learning
    R, data = reduction_to_gamma(data=data,
                                 alpha=alpha_bayes,
                                 M=M_bayes_1,
                                 intervals=intervals,
                                 replacement=replacement,
                                 target=target,
                                 gamma=gamma,
                                 eps=eps)

    if len(R) > 13:
        flag = True

        coins = R.flatten()  # new coins
        # pad R with the extremes of the initial problem
        min_R = min(coins)
        max_R = max(coins)
        # add to first element [0, min_R] and last element [max_R, B]
        R = np.insert(R, 0, np.array([intervals[0][0], min_R]), axis=0)
        R = np.insert(R, len(R), [max_R, intervals[-1][1]], axis=0)

        gamma = 1 / 13
        R, data = reduction_to_gamma(data=data,
                                     alpha=alpha_bayes,
                                     M=M_bayes_2,
                                     intervals=R,
                                     replacement=replacement,
                                     target=target,
                                     gamma=gamma,
                                     eps=eps)

    if flag:
        # if the second Bayesian learning was needed, use the remaining samples for the sampling
        M_sampling = M - M_bayes_1 - M_bayes_2
    else:
        M_sampling = M - M_bayes_1

    if naive_NBS:
        good_coin = naive_noisy_binary_search(data=data,
                                              intervals=R,
                                              M=M_sampling,
                                              eps=eps,
                                              target=target,
                                              replacement=replacement)
    else:
        # get the empirical cdf of the coins
        coins_prob = toss_coins(M=M_sampling, intervals=R, D=data, replacement=replacement, eps=eps)

        # get unbiased estimate of the empirical cdf
        factor_1 = (np.exp(eps) + 1) / (np.exp(eps) - 1)
        factor_2 = 1 / (np.exp(eps) + 1)
        coins_prob = {coin: factor_1 * (prob - factor_2) for coin, prob in coins_prob.items()}

        # check if a coin has a cdf smaller tha tau + alpha and bigger than tau - alpha
        good_coin = check_coins(alpha=alpha, alpha_bayes=alpha_bayes, intervals=R, coins_prob=coins_prob)

    return good_coin


def reduction_to_gamma(data: list,
                       alpha: float,
                       eps: float,
                       M: int,
                       intervals: np.array,
                       gamma: float,
                       replacement: bool,
                       target: float = 0.5) -> tuple[np.array, list]:
    """
    Reduction to gamma algorithm. It uses Bayesian learning to reduce the number of intervals.

    :param data: dataset of values in [B]
    :param alpha: target error
    :param eps: privacy parameter
    :param M: number of steps (samples)
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param gamma: fraction for the reduction
    :param replacement: with (True) / without (False) replacement
    :param target: target or quantile in (0,1) value

    :return: R: list of intervals
    :return: D: the dataset after the sampling
    """
    # run Bayesian learning
    L, data = bayes_learn(data=data,
                          alpha=alpha,
                          eps=eps,
                          M=M,
                          intervals=intervals,
                          replacement=replacement,
                          target=target)

    # reduction to gamma
    quantiles = [gamma * i for i in range(1, (int(np.floor(1 / gamma)) + 1))]
    R = np.zeros(len(quantiles), dtype=int)
    for i, quantile in enumerate(quantiles):
        R[i] = int(round(np.quantile(L, quantile)))
    # remove duplicates from R
    R = list(set(R))
    R.sort()
    R = intervals[R]
    return R, data


def toss_coins(M: int,
               intervals: list,
               D: list,
               eps: float,
               replacement: bool) -> dict[int, float]:
    """
    Toss coins and update the histogram
    :param M: number of tosses
    :param intervals: list of the intervals
    :param D: dataset to sample
    :param eps: privacy parameter
    :param replacement: with (True) / without (False) replacement
    :return: empirical cdf of the coins
    """
    # use M_sampling to sample more coins and update the histogram
    M_sampling_coin = M // len(intervals)

    # generate bernoulli coins with probability np.exp(eps) / (1 + np.exp(eps)) for Randomized Response
    random_coin = np.random.binomial(1, np.exp(eps) / (1 + np.exp(eps)), (len(intervals), M_sampling_coin))

    # sample the users
    samples = np.random.choice(D, (len(intervals), M_sampling_coin), replace=replacement)

    # generate the coins (only the right coin is needed)
    right_coins = intervals[:, 1].reshape(-1, 1)

    # flip the coins (in a vectorized way)
    y = samples <= right_coins

    # apply randomized response to the coin flip
    y = vector_RR(y, random_coin)

    # get the empirical cdf of the coins
    coins_prob = defaultdict(lambda: 0)
    for i, right_coin in enumerate(right_coins.flatten()):
        coins_prob[right_coin] = np.sum(y[i]) / M_sampling_coin
    return coins_prob
