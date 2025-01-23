import sys
import os
import pickle
import tqdm
import argparse

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
from hierarchical_mechanism.mechanism import hierarchical_mechanism_quantile
from hierarchical_mechanism.data_structure import Tree

# ------------Parameters of the data------------#

"""
    Select here the parameters of the experiment, for our experiments we used:
    - B_exp = 8, 9
    - N = 2500, 5000, 7500
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int,
                        help='Number of samples', required=True)
    parser.add_argument('--B_exp', type=int,
                        help='Exponent of the number of bins', required=True)
    return parser.parse_args()


args = parse_arguments()
B_exp = args.B_exp  # exponent of the number of bins 4^B_exp
N = args.N  # number of data points
folder_name = f"data/N_{N}/B_exp_{B_exp}"

# import data
with open(f'{folder_name}/pareto_data.pkl', 'rb') as f:
    data = pickle.load(f)

# import bins (coins)
with open(f'{folder_name}/pareto_bins.pkl', 'rb') as f:
    bins = pickle.load(f)

# ------------Parameters of the mechanism------------#
eps_list = np.geomspace(0.1, 5, 10)  # list of privacy budgets
target = 0.5
replacement = False  # sample without replacement
num_exp = 200  # number of experiments

# ------------Hierarchical Mechanism------------#
print("Hierarchical Mechanism")
print(f"Number of experiments: {num_exp}")
print(f"Number of data points: {N}")
print(f"B_exp: {B_exp}")
coins = np.zeros((len(eps_list), num_exp))  # store the output of the mechanism (epsilon, experiment)

# instantiating the tree data structure
tree = Tree(bins, branching=4)
for i, eps in tqdm.tqdm(enumerate(eps_list), total=len(eps_list), colour="green"):
    for j in range(num_exp):
        coin = hierarchical_mechanism_quantile(tree=tree,
                                               data=data,
                                               protocol="unary_encoding",
                                               eps=eps,
                                               target=target,
                                               replacement=replacement)
        coins[i, j] = coin

# save results
folder_name = f"results/hierarchical_mechanism/N_{N}/B_exp_{B_exp}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
