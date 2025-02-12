import numpy as np
from scipy.stats import ecdf
import matplotlib.pyplot as plt
import pickle
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='the seed used to generate random numbers (default: 42)')
    parser.add_argument('--N', type=int,
                        help='Number of samples', required=True)
    parser.add_argument('--B', type=int,
                        help='Exponent of the number of bins', required=True)
    return parser.parse_args()

def generate_random_center_uniform(N, B):
    # sample a random left endpoint
    left = np.random.choice(range(0, int(B / 2)), 1)[0].astype(int)
    # sample a random right endpoint
    right = np.random.choice(range(int(B / 2), B), 1)[0].astype(int)
    return np.random.choice(range(left, right + 1), N).astype(int)


"""
    Select here the parameters for generating the data, we used:
    - N = 2500
    - B = 10^3, 10^4, 10^5, 10^6, 10^7
"""

args = parse_arguments()
seed = args.seed
N = args.N  # number of samples
B = int(args.B)  # number of bins

np.random.seed(seed)

# generate the data
data = generate_random_center_uniform(N, B)

# get the empirical cdf of the coins
cf = ecdf(data)
cf_dict = dict(zip(cf.cdf.quantiles, cf.cdf.probabilities))
cf_dict[0] = 0
cf_dict[B] = 1
# sort by key
cf_dict = dict(sorted(cf_dict.items()))

# get the median quantile
median = None
for i, j in enumerate(cf_dict.keys()):
    if cf_dict[j] > 0.5:
        median = int(list(cf_dict.keys())[i - 1])
        break
median_quantile = cf_dict[median]

# create folder in data/data_{N}
folder_name = f"data/N_{N}/B_{B}"
os.makedirs(f"{folder_name}", exist_ok=True)

# plot cdf
fig, ax = plt.subplots()
plt.plot(cf_dict.keys(), cf_dict.values(), label=f"Median quantile: {median_quantile:.2f} and median: {median}")
plt.xlabel("data")
plt.xscale("log")
plt.ylabel("cdf")
plt.title("Empirical Sensitive CDF")
plt.legend()
# save figure
plt.savefig(f"{folder_name}/cdf.png")

# plot histogram
fig, ax = plt.subplots()
plt.hist(data, bins=100, density=True)
plt.xlabel("data")
plt.ylabel("density")
plt.title("Empirical Sensitive Histogram")
# save figure
plt.savefig(f"{folder_name}/hist.png")

# save data
with open(f"{folder_name}/data.pkl", "wb") as f:
    pickle.dump(data, f)

# save cdf
with open(f"{folder_name}/cdf.pkl", "wb") as f:
    pickle.dump(cf_dict, f)

# save median
with open(f"{folder_name}/median.pkl", "wb") as f:
    pickle.dump(median, f)

# save median quantile
with open(f"{folder_name}/median_quantile.pkl", "wb") as f:
    pickle.dump(median_quantile, f)
