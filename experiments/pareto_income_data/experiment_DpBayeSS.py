import sys
import os
import pickle
import tqdm
import argparse

import numpy as np

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


def get_th_alpha(B: int, N: int, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (np.sqrt(N))


# ------------Parameters of the data------------#

"""
    Select here the parameters of the experiment, for our experiments we used:
    - B_exp = 8, 9
    - N = 2500, 5000, 7500
    - c = 0.6
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int,
                        help='Number of samples', required=True)
    parser.add_argument('--B_exp', type=int,
                        help='Exponent of the number of bins', required=True)
    parser.add_argument('--c', type=float,
                        help='c parameter', required=True)
    parser.add_argument('--num-exp', type=int,
                        help='Number of experiments', required=False, default=200)
    return parser.parse_args()


args = parse_arguments()
B_exp = args.B_exp  # exponent of the number of bins 4^B_exp
N = args.N  # number of data points
c = args.c  # multiplicative factor for theoretical alpha = O(sqrt(log(B))/sqrt(N))

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

# ------------Parameters of the mechanism------------#
eps_list = np.geomspace(0.1, 5, 10)  # list of privacy budgets
target = 0.5
replacement = False  # sample without replacement
num_exp = args.num_exp  # number of experiments

# ------------BaySS------------#
print("BaySS")
print("c", c)
print("N", N)
print("B_exp", B_exp)
coins = np.zeros((len(eps_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
alpha = get_th_alpha(B=4 ** B_exp, N=N, c=c)
for i, eps in tqdm.tqdm(enumerate(eps_list), total=len(eps_list), colour="green"):
    for j in range(num_exp):
        coin = bayss_dp(data=data,
                        intervals=intervals,
                        alpha_update=alpha,
                        eps=eps,
                        target=target,
                        replacement=replacement)
        coins[i, j] = coin

# save results
folder_name = f"results/BayeSS_optimized/N_{N}/B_exp_{B_exp}/c_{c:.2f}".replace('.', '')
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
