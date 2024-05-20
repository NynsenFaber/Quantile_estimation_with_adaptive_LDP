import sys
import os
import pickle
import tqdm

import numpy as np
from utils import check_coin

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory
grandparent_dir = os.path.dirname(parent_dir)
# Add both parent and grandparent directories to the Python path
sys.path.append(grandparent_dir)

# Import the required module from gretta_price_dp
from BaySS.mechanism import bayss_dp


def upload_data(N: int, B_exp: int):
    folder_name = f"data/N_{N}/B_exp_{B_exp}"
    output = {}
    # import data
    with open(f'{folder_name}/pareto_data.pkl', 'rb') as f:
        data = pickle.load(f)
    output["data"] = data

    # import bins
    with open(f'{folder_name}/pareto_bins.pkl', 'rb') as f:
        bins = pickle.load(f)
    output["bins"] = bins

    # import intervals
    with open(f'{folder_name}/pareto_intervals.pkl', 'rb') as f:
        intervals = pickle.load(f)
    output["intervals"] = intervals

    # import median
    with open(f'{folder_name}/pareto_median.pkl', 'rb') as f:
        median = pickle.load(f)
    output["median"] = median

    # import median quantile
    with open(f'{folder_name}/pareto_median_quantile.pkl', 'rb') as f:
        median_quantile = pickle.load(f)
    output["median_quantile"] = median_quantile

    # import cdf
    with open(f'{folder_name}/pareto_cdf.pkl', 'rb') as f:
        cf_dict = pickle.load(f)
    output["cf_dict"] = cf_dict

    return output


def get_th_alpha(B: int, N: int, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (np.sqrt(N))


# ------------Parameters of the data------------#
eps = 1.5
N = 2500
B_exp = 9
num_exp = 200
alpha_test = 0.01

# ------------Parameters of the mechanism------------#
data_dict = upload_data(N, B_exp)
c_list = np.linspace(0.1, 2, 10)
target = 0.5
replacement = False
tries = 1

print("--- Find alpha update ---")
print("N", N)
print("B_exp", B_exp)
print("eps", eps)
print("alpha_test", alpha_test)
print("c_list", c_list)
print("num_exp", num_exp)

coins = np.zeros((len(c_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(c_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(c_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
alphas = np.zeros(len(c_list))  # store the alpha of the mechanism (epsilon, experiment)
for i, c in tqdm.tqdm(enumerate(c_list)):
    alpha = get_th_alpha(4 ** B_exp, N, c)
    print("Searching for alpha: ", alpha)
    alphas[i] = alpha
    for j in range(num_exp):
        coin = bayss_dp(data=data_dict["data"],
                        intervals=data_dict["intervals"],
                        M=len(data_dict["data"]),
                        alpha=alpha,
                        eps=eps,
                        target=target,
                        replacement=replacement,
                        naive_NBS=True)
        succ, err = check_coin(coin=coin, cf_dict=data_dict["cf_dict"], target=target, alpha=alpha_test,
                               median=data_dict["median"])
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/BaySS/find_alpha_{tries}/N_{N}/B_exp_{B_exp}/eps_{eps}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)
with open(f"{folder_name}/alphas.pkl", "wb") as f:
    pickle.dump(alphas, f)
with open(f"{folder_name}/c_list.pkl", "wb") as f:
    pickle.dump(c_list, f)
