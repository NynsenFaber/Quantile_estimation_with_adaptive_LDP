import numpy as np
import time
from .bayesian_learn_median import bayes_learn


def median_bayesian_screening_search(D: list,
                                     alpha: float,
                                     eps: float,
                                     M: int,
                                     intervals: np.array,
                                     replacement: bool = False) -> int:
    """
    Algorithm developed by Gretta-Price. It uses Bayesian learning to reduce the number of intervals. This version works
    with differential privacy and sampling without replacement.

    Ref: Gretta, Lucas, and Eric Price. "Sharp Noisy Binary Search with Monotonic Probabilities." arXiv preprint arXiv:2311.00840 (2023).
    :param D: dataset of values in [B]
    :param alpha: target error
    :param eps: privacy parameter
    :param M: steps (samples)
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param replacement: with (True) / without (False) replacement

    :return: the median
    """

    B = len(intervals) + 1  # number of coins

    # get the number of samples at for each step of the Bayesian learning
    M_bayes = int(M * np.log(B) / (np.log(B) + 1))
    M_bayes_1 = int(M_bayes * np.log(B) / (np.log(B) + 1))
    M_bayes_2 = M_bayes - M_bayes_1
    flag = False  # to keep track if a second Bayesian learning is needed

    # define the alpha for the Bayesian learning and the gamma for the reduction
    alpha_bayes = alpha * (2 / 3)
    gamma = 1 / (np.log(B) ** 2)  # as proposed in the reference paper

    # get the intervals from the Bayesian learning
    R, D = reduction_to_gamma(D, alpha_bayes, M_bayes_1, intervals, replacement, gamma, eps)
    if len(R) > 13:
        # print(f"Bayesian learning found {len(R)} intervals, running a second Bayesian learning")
        flag = True

        coins = R.flatten()
        # pad R with the extremes of the initial problem
        min_R = min(coins)
        max_R = max(coins)
        # add to first element [0, min_R] and last element [max_R, B]
        R = np.insert(R, 0, np.array([0, min_R]), axis=0)
        R = np.insert(R, len(R), [max_R, intervals[-1][1]], axis=0)

        gamma = 1 / 13
        R, D, L, H = reduction_to_gamma(D, alpha_bayes, M_bayes_2, R, replacement, gamma, eps)

    if flag:
        M_sampling = M - M_bayes_1 - M_bayes_2
    else:
        M_sampling = M - M_bayes_1
    assert M == M_bayes + M_sampling

    # get the coins from the set of intervals
    coins = list(set(R.flatten()))

    if not get_previous_coin:
        H = defaultdict(lambda: [0, 0])

    # get the empirical cdf of the coins
    coins_prob = toss_coins(M=M_sampling, coins=coins, H=H, D=D, replacement=replacement)

    # check if a coin has a cdf smaller tha tau + alpha and bigger than tau - alpha
    good_coin = check_coins(alpha=alpha, alpha_bayes=alpha_bayes, coins=coins, coins_prob=coins_prob)

    return R, L, coins_prob, good_coin


def reduction_to_gamma(D: list,
                       alpha: float,
                       eps: float,
                       M: int,
                       intervals: np.array,
                       gamma : float,
                       replacement: bool) -> tuple[np.array, list]:
    """
    Reduction to gamma algorithm. It uses Bayesian learning to reduce the number of intervals.
    :param D: dataset of values in [B]
    :param alpha: target error
    :param eps: privacy parameter
    :param M: number of steps (samples)
    :param intervals: list of intervals [[x_1, x_2], [x_2, x_3], ...]
    :param gamma: fraction for the reduction
    :param replacement: with (True) / without (False) replacement

    :return: R: list of intervals
    :return: D: the dataset after the sampling
    """
    # run Bayesian learning
    start = time.time()
    L, D = bayes_learn(D=D, alpha=alpha, eps=eps, M=M, intervals=intervals, replacement=replacement)
    # print(f"Bayesian learning took {time.time() - start} seconds")

    # reduction to gamma
    start = time.time()
    quantiles = [gamma * i for i in range(1, (int(np.floor(1 / gamma)) + 1))]
    R = np.zeros(len(quantiles))
    for i, quantile in enumerate(quantiles):
        R[i] = int(round(np.quantile(L, quantile)))
    # remove duplicates from R
    R = list(set(R))
    # R must be integers
    R = [int(r) for r in R]
    R.sort()
    # print(f"Reduction to gamma took {time.time() - start} seconds")
    R = intervals[R]
    return R, D