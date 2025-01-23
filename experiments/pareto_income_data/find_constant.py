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


def upload_data(N: int, B_exp: int):
    folder = f"data/N_{N}/B_exp_{B_exp}"
    output = {}
    # import data
    with open(f'{folder}/pareto_data.pkl', 'rb') as f:
        data = pickle.load(f)
    output["data"] = data

    # import intervals
    with open(f'{folder}/pareto_intervals.pkl', 'rb') as f:
        intervals = pickle.load(f)
    output["intervals"] = intervals

    return output


def get_th_alpha(B: int, N: int, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (np.sqrt(N))


# ------------Parameters of the data------------#

"""
    for our experiments we used:
    - B_exp = 8, 9
    - N = 2500, 5000
    - eps = 0.5, 1, 1.5
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int,
                        help='Number of samples', required=True)
    parser.add_argument('--B_exp', type=int,
                        help='Exponent of the number of bins', required=True)
    parser.add_argument('--eps', type=float,
                        help='privacy budgest', required=True)
    return parser.parse_args()


args = parse_arguments()

eps = args.eps  # privacy budget
N = args.N  # number of data points
B_exp = args.B_exp  # exponent of the number of bins
num_exp = 200  # number of experiments

# ------------Parameters of the mechanism------------#
data_dict = upload_data(N, B_exp)
c_list = np.linspace(0.1, 2, 10)
target = 0.5
replacement = False

print("--- Find alpha update ---")
print("N", N)
print("B_exp", B_exp)
print("eps", eps)
print("c_list", c_list)
print("num_exp", num_exp)

coins = np.zeros((len(c_list), num_exp))  # store the output of the mechanism (epsilon, experiment)
for i, c in tqdm.tqdm(enumerate(c_list), total=len(c_list), colour="green"):
    alpha = get_th_alpha(4 ** B_exp, N, c)
    for j in range(num_exp):
        coin = bayss_dp(data=data_dict["data"],
                        intervals=data_dict["intervals"],
                        alpha_update=alpha,
                        eps=eps,
                        target=target,
                        replacement=replacement)
        coins[i, j] = coin

# save results
folder_name = f"results/BaySS_find_constant/N_{N}/B_exp_{B_exp}/eps_{eps}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(coins, f)
with open(f"{folder_name}/c_list.pkl", "wb") as f:
    pickle.dump(c_list, f)
