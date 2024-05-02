import numpy as np
from .BayesLearn import median_bayes_learn, sample_element, flip_coin
from .utils import get_coins
from collections import defaultdict
import time


def median_bayesian_screening_search(D: np.array,
                                     alpha: float,
                                     M: int,
                                     intervals: np.array,
                                     replacement: bool = False,
                                     get_previous_coin: bool = False) -> np.array:
    """
    Algorithm developed by Gretta-Price
    :param D:
    :param alpha:
    :param M:
    :param intervals:
    :param replacement:
    :param get_previous_coin:
    :return:
    """
    B = len(intervals) + 1
    M_bayes = int(M * np.log(B) / (np.log(B) + 1))
    M_bayes_1 = int(M_bayes * np.log(B) / (np.log(B) + 1))
    M_bayes_2 = M_bayes - M_bayes_1

    flag = False

    # print(f"Bayesian learning will use {M_bayes} samples and sampling will use {M_sampling} samples")

    alpha_bayes = alpha * (2 / 3)
    gamma = 1 / (np.log(B) ** 2)

    # get the intervals from the Bayesian learning
    R, D, L, H = reduction_to_gamma(D, alpha_bayes, M_bayes_1, intervals, replacement, gamma)
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
        R, D, L, H = reduction_to_gamma(D, alpha_bayes, M_bayes_2, R, replacement, gamma)

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


def reduction_to_gamma(D, alpha, M, intervals, replacement, gamma):

    # run Bayesian learning
    start = time.time()
    L, H, D = median_bayes_learn(D=D, alpha=alpha, M=M, intervals=intervals, replacement=replacement)
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
    return R, D, L, H  # remove duplicates


def toss_coins(M: int, coins: list,
               H: dict[int, tuple[int, int]],
               D: list,
               replacement: bool) -> dict[int, float]:
    """
    Toss coins and update the histogram
    :param M: number of tosses
    :param coins: list of coins to toss
    :param replacement: with (True) / without (False) replacement
    :param D: dataset to sample
    :param H: dictionary containing coins as keys and a tuple[#tosses, #heads] as values
    :return: empirical cdf of the coins
    """
    # use M_sampling to sample more coins and update the histogram
    M_sampling_coin = M // len(coins)
    for coin in coins:
        for _ in range(M_sampling_coin):
            sample = sample_element(D=D, replacement=replacement)
            y = flip_coin(coin=coin, sample=sample)
            H[coin][0] += 1
            H[coin][1] += y
    # # add missing coins to the histogram
    # missing_coins = set(coins) - set(H.keys())
    # for missing_coin in missing_coins:
    #     H[missing_coin] = [0, 0]
    # get the empirical cdf of the coins
    coins_prob = defaultdict(lambda: 0)
    for coin, (tosses, heads) in H.items():
        if tosses == 0:
            coins_prob[coin] = 0
        else:
            coins_prob[coin] = heads / tosses
    return coins_prob


def check_coins(alpha: float, alpha_bayes: float, coins: list, coins_prob: dict[int, float]) -> int:
    """
    Check if a coin is good or not, return a random good coin
    :param alpha: error to seek
    :param alpha_bayes: error for the Bayesian learning
    :param coins: list of coins to check
    :param coins_prob: empirical cdf of the coins
    :return: a random good coin
    """
    good_coins = []
    left_interval = 0.5 + alpha - (alpha - alpha_bayes) / 2
    right_interval = 0.5 - alpha + (alpha - alpha_bayes) / 2
    for coin in coins:
        if coins_prob[coin] < right_interval or coins_prob[coin] > left_interval:
            # print(f"Coin {coin} has a ecdf {coins_prob[coin]} outside the interval [{left_interval}, {right_interval}]")
            continue
        else:
            # print(
            #     f"GOOD COIN ---> Coin {coin} has a ecdf {coins_prob[coin]} inside the interval [{left_interval}, {right_interval}]")
            good_coins.append(coin)
    if len(good_coins) > 0:
        random_good_coin = np.random.choice(good_coins, 1)[0]
    else:
        # randomly select a coin
        # print("No good coins found, selecting a random coin from the quantiles")
        random_good_coin = np.random.choice(list(coins), 1)[0]
    return random_good_coin
