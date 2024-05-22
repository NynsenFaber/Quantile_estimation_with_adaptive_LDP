import numpy as np
from scipy.stats import ecdf
import matplotlib.pyplot as plt
import pickle
import os


def generate_random_center_uniform(N, B):
    # sample a random left endpoint
    left = np.random.choice(range(0, int(B / 2)), 1)[0].astype(int)
    # sample a random right endpoint
    right = np.random.choice(range(int(B / 2), B), 1)[0].astype(int)
    print(left, right)
    return np.random.choice(range(left, right + 1), N).astype(int)


seed = 42
N = 2500  # number of samples
B = int(1E7)

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
