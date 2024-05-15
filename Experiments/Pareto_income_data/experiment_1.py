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

# ------------Parameters of the data------------#
B_exp = 9
N = 5000
folder_name = f"data/N_{N}/B_exp_{B_exp}"

# import data
with open(f'{folder_name}/pareto_data.pkl', 'rb') as f:
    data = pickle.load(f)

# import bins
with open(f'{folder_name}/pareto_bins.pkl', 'rb') as f:
    bins = pickle.load(f)

# import intervals
with open(f'{folder_name}/pareto_intervals.pkl', 'rb') as f:
    intervals = pickle.load(f)

# import median
with open(f'{folder_name}/pareto_median.pkl', 'rb') as f:
    median = pickle.load(f)

# import median quantile
with open(f'{folder_name}/pareto_median_quantile.pkl', 'rb') as f:
    median_quantile = pickle.load(f)

# import cdf
with open(f'{folder_name}/pareto_cdf.pkl', 'rb') as f:
    cf_dict = pickle.load(f)

# ------------Parameters of the mechanism------------#
eps_list = np.geomspace(0.1, 5, 10)
target = 0.5
alpha = 0.05
replacement = False
num_exp = 200

# ------------Noisy Binary Search------------#
print("Noisy Binary Search")
coins = np.zeros((len(eps_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(eps_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(eps_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
for i, eps in tqdm.tqdm(enumerate(eps_list)):
    for j in range(num_exp):
        coin = noisy_binary_search(data=data,
                                   intervals=intervals,
                                   M=len(data),
                                   alpha=alpha,
                                   eps=eps,
                                   target=target,
                                   replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=cf_dict, target=target, alpha=alpha, median=median)
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/noisy_binary_search/N_{N}/B_exp_{B_exp}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)

# ------------Gretta Price------------#
print("Gretta Price")
coins = np.zeros((len(eps_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(eps_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(eps_list), num_exp))  # store the success of the mechanism (epsilon, experiment)
for i, eps in tqdm.tqdm(enumerate(eps_list)):
    for j in range(num_exp):
        coin = gretta_price_dp(data=data,
                               intervals=intervals,
                               M=len(data),
                               alpha=alpha,
                               eps=eps,
                               target=target,
                               replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=cf_dict, target=target, alpha=alpha, median=median)
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/gretta_price/N_{N}/B_exp_{B_exp}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)

# ------------Hierarchical Mechanism------------#
print("Hierarchical Mechanism")
coins = np.zeros((len(eps_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
errors = np.zeros((len(eps_list), num_exp))  # store the error of the mechanism (epsilon, experiment)
success = np.zeros((len(eps_list), num_exp))  # store the success of the mechanism (epsilon, experiment)

# instantiating the tree data structure
tree = Tree(bins, branching=4)
for i, eps in tqdm.tqdm(enumerate(eps_list)):
    for j in range(num_exp):
        coin = hierarchical_mechanism_quantile(tree=tree,
                                               data=data,
                                               protocol="unary_encoding",
                                               eps=eps,
                                               target=target,
                                               replacement=replacement)
        succ, err = check_coin(coin=coin, cf_dict=cf_dict, target=target, alpha=alpha, median=median)
        coins[i, j] = coin
        errors[i, j] = err
        success[i, j] = succ

# save results
folder_name = f"results/hierarchical_mechanism/N_{N}/B_exp_{B_exp}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/errors.pkl", "wb") as f:
    pickle.dump(errors, f)
with open(f"{folder_name}/success.pkl", "wb") as f:
    pickle.dump(success, f)
