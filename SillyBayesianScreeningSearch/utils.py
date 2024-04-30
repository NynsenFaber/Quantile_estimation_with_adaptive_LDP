import math


def binary_entropy(p: float) -> float:
    """
    Calculate the binary entropy of a Bernoulli random variable with probability p.
    :param p: probability of the Bernoulli random variable
    :return: binary entropy of the Bernoulli random variable
    """
    if p == 0 or p == 1:
        return 0
    else:
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def get_z(tau: float, alpha: float) -> float:
    exp = (binary_entropy(tau - alpha) - binary_entropy(tau + alpha)) / (2 * alpha)
    return 2 ** exp


def get_q(tau: float, alpha: float) -> float:
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1

    num = (1 - tau - alpha) - 1 / (1 + get_z(tau, alpha))
    return num / (2 * alpha)


def get_BAC_capacity(tau: float, alpha: float) -> float:
    """
    Calculate the capacity of the Binary Asymmetric Channel (BAC) with crossover probability tau and asymmetry alpha.
    :param tau: crossover probability
    :param alpha: asymmetry
    :return: Capacity of the BAC
    """
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1

    add_1 = math.log2(1 + get_z(tau, alpha))
    add_2 = (tau - alpha) / (2 * alpha) * binary_entropy(tau + alpha)
    add_3 = -(tau + alpha) / (2 * alpha) * binary_entropy(tau - alpha)
    return add_1 + add_2 + add_3


def get_d00(tau: float, alpha: float) -> float:
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1

    q = get_q(tau, alpha)
    num = 1 - tau - alpha
    den = 1 - tau - (2 * q - 1) * alpha
    return num / den


def get_d01(tau: float, alpha: float) -> float:
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1

    q = get_q(tau, alpha)
    num = 1 - tau + alpha
    den = 1 - tau - (2 * q - 1) * alpha
    return num / den


def get_d10(tau: float, alpha: float) -> float:
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1

    q = get_q(tau, alpha)
    num = tau + alpha
    den = tau + (2 * q - 1) * alpha
    return num / den


def get_d11(tau: float, alpha: float) -> float:
    assert 0 <= tau <= 1
    assert 0 <= alpha <= 1
    assert tau - alpha >= 0
    assert tau + alpha <= 1

    q = get_q(tau, alpha)
    num = tau - alpha
    den = tau + (2 * q - 1) * alpha
    return num / den
