def check_coin(coin: int, cf_dict: dict, target: float, alpha: float, median: int) -> tuple[bool, float]:
    if coin not in cf_dict.keys():
        for index in reversed(range(coin)):
            if index in cf_dict.keys():
                coin = index
                break
    if target - alpha < cf_dict[coin] < target + alpha:
        success = True
    else:
        success = False
    error = cf_dict[coin] - cf_dict[median]
    return success, error
