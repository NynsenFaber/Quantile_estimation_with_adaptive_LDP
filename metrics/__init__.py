def get_quantile(value: float, cdf: dict[int, float]) -> float:
    """
    Get the quantile given a cumulative distribution function

    :param value: float, value to find the quantile for
    :param cdf: dict[int, float], cumulative distribution function

    :return: float, quantile
    """
    output = 0
    if value in cdf.keys():
        output = cdf[value]
    else:  # search for the closest quantile
        for quantile in reversed(range(value)):
            if quantile in cdf.keys():
                output = cdf[quantile]
                break
    return output


def get_quantile_absolute_error(x: float, y: float, cdf: dict[int, float]) -> float:
    """
    Get the absolute error between two values

    :param x: float, first value
    :param y: float, second value
    :param cdf: dict[int, float], cumulative distribution function

    :return: float, absolute error
    """
    return abs(get_quantile(x, cdf) - get_quantile(y, cdf))


def get_success(value: float, alpha: float, cdf: dict[int, float], tau: float = 0.5) -> bool:
    """
    Check if a quantile is alpha-good

    :param value: float, value to check
    :param alpha: float, threshold
    :param cdf: dict[int, float], cumulative distribution function
    :param tau: float, optimal quantile

    :return: bool, True if success, False otherwise
    """
    assert 0 < alpha < 1, "Alpha must be between 0 and 1"
    quantile_left = get_quantile(value, cdf)
    quantile_right = get_quantile(value + 1, cdf)
    return quantile_left < tau + alpha and quantile_right > tau - alpha
