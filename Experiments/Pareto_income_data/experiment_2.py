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
from gretta_price_dp.mechanism import gretta_price_dp
from noisy_binary_search.mechanism import noisy_binary_search
from hierarchical_mechanism.mechanism import hierarchical_mechanism_quantile
from hierarchical_mechanism.data_structure import Tree


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


def get_th_alpha(B: int, N: int, eps: float, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (eps * np.sqrt(N))


# ------------Parameters of the data------------#
eps = 1
N = 1000
c = 1  # multiplicative factor for theoretical alpha = O(sqrt(log(B))/eps*sqrt(N))

# ------------Parameters of the mechanism------------#
B_exp_list = [4, 5, 6, 7, 8, 9]
target = 0.5
replacement = False
num_exp = 2

# ------------Noisy Binary Search------------#
print("Noisy Binary Search")
coins = np.zeros((len(B_exp_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(B_exp_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(B_exp_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
for i, eps in tqdm.tqdm(enumerate(B_exp_list)):
    data_dict = upload_data(N=N, B_exp=eps)

    # get the theoretical alpha
    alpha = get_th_alpha(B=2 ** eps, N=N, eps=eps, c=c)

    # run experiments
    for j in range(num_exp):
        coin = noisy_binary_search(data=data_dict["data"],
                                   intervals=data_dict["intervals"],
                                   M=len(data_dict["data"]),
                                   alpha=alpha,
                                   eps=eps,
                                   target=target,
                                   replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=data_dict["cf_dict"], target=target, alpha=alpha,
                               median=data_dict["median"])
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/noisy_binary_search/N_{N}/eps_{eps}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)

# ------------Gretta Price------------#
print("Gretta Price")
coins = np.zeros((len(B_exp_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(B_exp_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(B_exp_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
for i, eps in tqdm.tqdm(enumerate(B_exp_list)):
    data_dict = upload_data(N=N, B_exp=eps)

    # get the theoretical alpha
    alpha = get_th_alpha(B=2 ** eps, N=N, eps=eps, c=c)

    for j in range(num_exp):
        coin = gretta_price_dp(data=data_dict["data"],
                               intervals=data_dict["intervals"],
                               M=len(data_dict["data"]),
                               alpha=alpha,
                               eps=eps,
                               target=target,
                               replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=data_dict["cf_dict"], target=target, alpha=alpha,
                               median=data_dict["median"])
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/gretta_price/N_{N}/eps_{eps}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)

# ------------Hierarchical Mechanism------------#
print("Hierarchical Mechanism")
coins = np.zeros((len(B_exp_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(B_exp_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(B_exp_list), num_exp))  # store the success of the mechanism (epsilon, experiment)

for i, eps in tqdm.tqdm(enumerate(B_exp_list)):
    data_dict = upload_data(N=N, B_exp=eps)

    # instantiating the tree data structure
    tree = Tree(data_dict["bins"], branching=4)

    # get the theoretical alpha
    alpha = get_th_alpha(B=2 ** eps, N=N, eps=eps, c=c)

    for j in range(num_exp):
        coin = hierarchical_mechanism_quantile(tree=tree,
                                               data=data_dict["data"],
                                               protocol="unary_encoding",
                                               eps=eps,
                                               target=target,
                                               replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=data_dict["cf_dict"], target=target, alpha=alpha,
                               median=data_dict["median"])
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/hierarchical_mechanism/N_{N}/eps_{eps}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)
