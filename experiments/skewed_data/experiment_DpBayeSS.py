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


def upload_data(N: int, B: int):
    folder_name = f"data/N_{N}/B_{B}"
    output = {}
    # import data
    with open(f'{folder_name}/data.pkl', 'rb') as f:
        data = pickle.load(f)
    output["data"] = data

    return output


def get_th_alpha(B: int, N: int, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (np.sqrt(N))


# ------------Parameters of the data------------#

"""
    Select here the parameters of the experiment, for our experiments we used:
    -eps = 0.5, 1
    -N = 2500
    -c = 0.6
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int,
                        help='Number of samples', required=True)
    parser.add_argument('--c', type=float,
                        help='c parameter', required=True)
    parser.add_argument('--eps', type=float,
                        help='privacy budget', required=True)
    return parser.parse_args()


args = parse_arguments()
eps = args.eps  # exponent of the number of bins 4^B_exp
N = args.N  # number of data points
c = args.c  # multiplicative factor for theoretical alpha = O(sqrt(log(B))/sqrt(N))

print("Second Experiment between Noisy Binary Search and DpBayeSS")
print(f"Number of data points: {N}")
print(f"Privacy parameter: {eps}")
print(f"Multiplicative factor for theoretical alpha: {c}")

alpha_test = 0.05

# ------------Parameters of the mechanism------------#
num_bins_list = [int(1E2), int(1E3), int(1E4), int(1E5), int(1E6)]  # list of number of bins
target = 0.5
replacement = False  # sample without replacement
num_exp = 200  # number of experiments

print(f"Number of experiments: {num_exp}")
print(f"Number of bins: {num_bins_list}")

# ------------DpBayeSS------------#
print("DpBayeSS")
coins = np.zeros((len(num_bins_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
for i, num_bins in tqdm.tqdm(enumerate(num_bins_list), total=len(num_bins_list), colour="green"):

    data_dict = upload_data(N=N, B=num_bins)

    bins = np.array(range(num_bins))  # Bin edges
    intervals = np.array([bins[:-1], bins[1:]]).T

    # get the theoretical alpha used in the mechanism
    alpha = get_th_alpha(B=num_bins, N=N, c=c)

    for j in range(num_exp):
        coin = bayss_dp(data=data_dict["data"],
                        intervals=intervals,
                        alpha_update=alpha,
                        eps=eps,
                        target=target,
                        replacement=replacement)
        coins[i, j] = coin

# save results
folder_name = f"results/BayeSS/N_{N}/eps_{eps}/bins_{int(num_bins_list[0])}_{int(num_bins_list[-1])}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/num_bins_list.pkl", "wb") as f:
    pickle.dump(num_bins_list, f)
