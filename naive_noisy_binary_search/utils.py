import numpy as np


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


def get_eps_for_shuffle(N: float, B: float, eps: float, delta: float) -> float:
    """
    Get the privacy parameter for the shuffle mechanism.
    :param N: number of users
    :param B: number of coins
    :param eps: privacy parameter
    :param delta: delta parameter

    :return: privacy budget epsilon for the shuffle mechanism
    """
    return np.log((eps ** 2 * N) / (80 * np.log(B) * np.log(4 / delta)))
