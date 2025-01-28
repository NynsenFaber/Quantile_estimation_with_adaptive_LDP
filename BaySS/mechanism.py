import numpy as np
import os
import sys
from .bayesian_learning import bayes_learn

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add parent directory to the Python path
sys.path.append(parent_dir)
from naive_noisy_binary_search.mechanism import naive_noisy_binary_search


def bayss_dp(data: list,
             alpha_update: float,
             eps: float,
             intervals: np.array,
             target: float = 0.5,
             replacement: bool = False,
             test: bool = False) -> int:
    """
    Algorithm developed by Gretta-Price. It uses Bayesian learning to reduce the number of intervals. This version works
    with differential privacy and sampling without replacement. It uses all the samples.
    It runs at most twice the Bayesian learning, then it uses noisy binary search.

    Ref:  Gretta, Lucas, and Eric Price. "Sharp Noisy Binary Search with Monotonic Probabilities."
          arXiv preprint arXiv:2311.00840 (2023).

    :param data: dataset of values in [B]
    :param alpha_update: error for the Bayesian learning
    :param eps: privacy parameter
    :param intervals: list of intervals [[c_1, c_2], [c_2, c_3], ...] containing the coins
    :param target: target or quantile in (0,1) value
    :param replacement: with (True) / without (False) replacement
    :param test: test flag

    :return: the median
    """

    data = list(data)  # data must be a list, so that it can be modified

    # get coins from intervals
    coins = list(set(intervals.flatten()))
    coins.sort()
    B = len(coins)
    M = len(data)  # number of samples

    # get the number of samples at for each step of the Bayesian learning
    # to have M_bayes_1/M_bayes_2 = log(B)/log(log(B)), M_bayes_1/M_sampling = log(B)
    # and M_bayes_2/M_sampling = log(log(B))
    den = np.log(B) + np.log(np.log(B)) + 1
    M_bayes_1 = int(M * np.log(B) / den)
    M_bayes_2 = int(M * np.log(np.log(B)) / den)

    flag = False  # to keep track if a second Bayesian learning is needed

    # alpha_bayes = alpha * (2 / 3)  # not used in the implementation of GP
    gamma = 1 / (np.log(B) ** 2)  # as proposed in the reference paper

    # get the intervals from the Bayesian learning
    initial_len = len(data)
    R, data = reduction_to_gamma(data=data,
                                 alpha_update=alpha_update,
                                 M=M_bayes_1,
                                 intervals=intervals,
                                 replacement=replacement,
                                 target=target,
                                 gamma=gamma,
                                 eps=eps)
    assert len(data) == initial_len - M_bayes_1, "The number of elements in the dataset must be equal to M "

    if len(R) > 13:
        # Apply again the Bayesian learning
        flag = True

        coins = R.flatten()  # new coins
        # pad R with the extremes of the initial problem
        min_R = min(coins)
        max_R = max(coins)
        # add to first element [0, min_R] and last element [max_R, B]
        R = np.insert(R, 0, np.array([intervals[0][0], min_R]), axis=0)
        R = np.insert(R, len(R), [max_R, intervals[-1][1]], axis=0)

        gamma = 1 / 13
        initial_len = len(data)
        R, data = reduction_to_gamma(data=data,
                                     alpha_update=alpha_update,
                                     M=M_bayes_2,
                                     intervals=R,
                                     replacement=replacement,
                                     target=target,
                                     gamma=gamma,
                                     eps=eps)
        assert len(data) == initial_len - M_bayes_2, "The number of elements in the dataset must be equal to M"

    if test:
        if flag:
            # if the second Bayesian learning was needed, use the remaining samples for the sampling
            M_sampling = M - M_bayes_1 - M_bayes_2
            assert len(data) == M_sampling, "The number of elements in the dataset must be equal to M"
        else:
            M_sampling = M - M_bayes_1
            assert len(data) == M_sampling, "The number of elements in the dataset must be equal to M"

    # get coins from the intervals R
    coins = list(set(R.flatten()))
    coins.sort()
    good_coin = naive_noisy_binary_search(data=np.array(data),
                                          coins=np.array(coins),
                                          eps=eps,
                                          target=target,
                                          replacement=replacement,
                                          test=test)

    return good_coin


def reduction_to_gamma(data: list,
                       alpha_update: float,
                       eps: float,
                       M: int,
                       intervals: np.array,
                       gamma: float,
                       replacement: bool,
                       target: float = 0.5) -> tuple[np.array, list]:
    """
    Reduction to gamma algorithm. It uses Bayesian learning to reduce the number of intervals.

    :param data: dataset of values in [B]
    :param alpha_update: error for the Bayesian learning
    :param eps: privacy parameter
    :param M: number of steps (samples)
    :param intervals: list of intervals [[c_1, c_2], [c_2, c_3], ...]
    :param gamma: fraction for the reduction
    :param replacement: with (True) / without (False) replacement
    :param target: target or quantile in (0,1) value

    :return: R: list of intervals
    :return: D: the dataset after the sampling
    """
    # run Bayesian learning
    L, data = bayes_learn(data=data,
                          alpha_update=alpha_update,
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
    R = list(set(R))      # remove duplicates from R indices
    R.sort()              # sort the indices
    R = intervals[R]      # get the intervals from the indices
    return R, data
