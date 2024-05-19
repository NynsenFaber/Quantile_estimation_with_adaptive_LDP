import numpy as np


def check_coins(alpha: float, coins: list, coins_prob: dict[int, float], target: float) -> int:
    """
    Check if a coin is good or not, return a random good coin. If no good coins are found, return a coin
    with probability closest to the target.
    :param alpha: error to seek
    :param coins: list of coins to check
    :param coins_prob: empirical cdf of the coins
    :param target: target or quantile in (0,1) value
    :return: a random good coin
    """
    good_coins = []
    right_bound = target + alpha
    left_bound = target - alpha
    for coin in coins:
        if coins_prob[coin] >= right_bound or coins_prob[coin] <= left_bound:
            continue
        else:
            good_coins.append(coin)
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
