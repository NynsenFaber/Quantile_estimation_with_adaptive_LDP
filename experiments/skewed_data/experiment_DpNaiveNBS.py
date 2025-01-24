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

from naive_noisy_binary_search.mechanism import naive_noisy_binary_search


def upload_data(N: int, B: int):
    folder = f"data/N_{N}/B_{B}"
    output = {}
    # import data
    with open(f'{folder}/data.pkl', 'rb') as f:
        data = pickle.load(f)
    output["data"] = data

    # import median
    with open(f'{folder}/median.pkl', 'rb') as f:
        median = pickle.load(f)
    output["median"] = median

    # import median quantile
    with open(f'{folder}/median_quantile.pkl', 'rb') as f:
        median_quantile = pickle.load(f)
    output["median_quantile"] = median_quantile

    # import cdf
    with open(f'{folder}/cdf.pkl', 'rb') as f:
        cf_dict = pickle.load(f)
    output["cf_dict"] = cf_dict

    return output


def get_th_alpha(B: int, N: int, c: float = 1) -> float:
    return c * np.sqrt(np.log(B)) / (np.sqrt(N))


# ------------Parameters of the data------------#

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int,
                        help='Number of samples', required=True)
    parser.add_argument('--eps', type=float,
                        help='privacy budgest', required=True)
    return parser.parse_args()


args = parse_arguments()
eps = args.eps  # privacy parameter
N = args.N  # number of data points

print("Second Experiment between Noisy Binary Search and DpBayeSS")
print(f"Number of data points: {N}")
print(f"Privacy parameter: {eps}")

# ------------Parameters of the mechanism------------#
num_bins_list = [int(1E2), int(1E3), int(1E4), int(1E5), int(1E6)]  # list of number of bins
target = 0.5
replacement = False  # sample without replacement
num_exp = 200  # number of experiments

print(f"Number of experiments: {num_exp}")
print(f"Number of bins: {num_bins_list}")

# ------------Noisy Binary Search------------#
print("Noisy Binary Search")
# store the output of the mechanism (B size of the domain, experiment)
returned_coins = np.zeros((len(num_bins_list), num_exp))
for i, num_bins in tqdm.tqdm(enumerate(num_bins_list), total=len(num_bins_list), colour="green"):
    # upload data
    data_dict = upload_data(N=N, B=num_bins)
    # get coins
    bins = np.array(range(num_bins))  # Bin edges
    # run experiments
    for j in range(num_exp):
        coin = naive_noisy_binary_search(data=data_dict["data"],
                                         coins=bins,
                                         eps=eps,
                                         target=target,
                                         replacement=replacement)
        returned_coins[i, j] = coin

# save results
folder_name = f"results/naive_noisy_binary_search/N_{N}/eps_{eps}/bins_{int(num_bins_list[0])}_{int(num_bins_list[-1])}"
os.makedirs(f"{folder_name}", exist_ok=True)

with open(f"{folder_name}/coins.pkl", "wb") as f:
    pickle.dump(returned_coins, f)
with open(f"{folder_name}/num_bins_list.pkl", "wb") as f:
    pickle.dump(num_bins_list, f)
