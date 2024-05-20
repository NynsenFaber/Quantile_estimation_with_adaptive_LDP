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
from naive_noisy_binary_search.mechanism import naive_noisy_binary_search


def upload_data(N: int):
    folder_name = f"data/continuum/N_{N}"
    output = {}
    # import data
    with open(f'{folder_name}/pareto_data.pkl', 'rb') as f:
        data = pickle.load(f)
    output["data"] = data

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
eps = 1
N = 2500
c = 0.6  # multiplicative factor for theoretical alpha = O(sqrt(log(B))/sqrt(N))

print("Second Experiment between Noisy Binary Search and DpBayeSS")
print(f"Number of data points: {N}")
print(f"Privacy parameter: {eps}")
print(f"Multiplicative factor for theoretical alpha: {c}")

alpha_test = 0.05

# ------------Parameters of the mechanism------------#
num_bins_list = np.linspace(10 ** 5, 10 ** 6, 10, dtype=int)
target = 0.5
replacement = False
num_exp = 200

print(f"Number of experiments: {num_exp}")
print(f"Number of bins: {num_bins_list}")

data_dict = upload_data(N=N)

# ------------Noisy Binary Search------------#
print("Noisy Binary Search")
coins = np.zeros((len(num_bins_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(num_bins_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(num_bins_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
for i, num_bins in tqdm.tqdm(enumerate(num_bins_list)):

    bins = np.array(range(num_bins))  # Bin edges
    intervals = np.array([bins[:-1], bins[1:]]).T

    # run experiments
    for j in range(num_exp):
        coin = naive_noisy_binary_search(data=data_dict["data"],
                                         intervals=intervals,
                                         M=len(data_dict["data"]),
                                         eps=eps,
                                         target=target,
                                         replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=data_dict["cf_dict"], target=target, alpha=alpha_test,
                               median=data_dict["median"])
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/continuum/noisy_binary_search/N_{N}/eps_{eps}/bins_{int(num_bins_list[0])}_{int(num_bins_list[-1])}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)
with open(f"{folder_name}/num_bins_list.pkl", "wb") as f:
    pickle.dump(num_bins_list, f)

# ------------DpBayeSS------------#
print("DpBayeSS")
coins = np.zeros((len(num_bins_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(num_bins_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(num_bins_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
for i, num_bins in tqdm.tqdm(enumerate(num_bins_list)):

    bins = np.array(range(num_bins))  # Bin edges
    intervals = np.array([bins[:-1], bins[1:]]).T

    # get the theoretical alpha
    alpha = get_th_alpha(B=num_bins, N=N, c=c)

    for j in range(num_exp):
        coin = bayss_dp(data=data_dict["data"],
                        intervals=intervals,
                        M=len(data_dict["data"]),
                        alpha=alpha,
                        eps=eps,
                        target=target,
                        replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=data_dict["cf_dict"], target=target, alpha=alpha_test,
                               median=data_dict["median"])
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/continuum/BayeSS/N_{N}/eps_{eps}/bins_{int(num_bins_list[0])}_{int(num_bins_list[-1])}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)
with open(f"{folder_name}/num_bins_list.pkl", "wb") as f:
    pickle.dump(num_bins_list, f)
