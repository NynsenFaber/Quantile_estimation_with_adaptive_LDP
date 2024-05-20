import numpy as np
from scipy.stats import ecdf
import matplotlib.pyplot as plt
import pickle
import os


def generate_pareto_data(shape, N, B_exp):
    if B_exp >= 8:
        D = np.random.pareto(shape, N) * 2000
    elif B_exp == 7:
        D = np.random.pareto(shape, N) * 500
    elif B_exp == 6:
        D = np.random.pareto(shape, N) * 200
    elif B_exp == 5:
        D = np.random.pareto(shape, N) * 50
    else:
        raise ValueError("B_exp must be greater than or equal to 5")
    return D


seed = 42
N = 2500  # number of samples
B_exp = 9

np.random.seed(seed)

# generate the data
data = generate_pareto_data(1.5, N, B_exp)

# discretize the data
num_bins = int(4 ** B_exp)  # Number of bins
bins = np.array(range(num_bins))  # Bin edges
intervals = np.array([bins[:-1], bins[1:]]).T
data = np.round(data).astype(int)  # integer data
data = np.clip(data, 0, num_bins - 1)

# get the empirical cdf of the coins
cf = ecdf(data)
cf_dict = dict(zip(cf.cdf.quantiles, cf.cdf.probabilities))

# get the median quantile
median = None
for i, j in enumerate(cf_dict.keys()):
    if cf_dict[j] > 0.5:
        median = int(list(cf_dict.keys())[i - 1])
        break
median_quantile = cf_dict[median]

# plot cdf
plt.plot(cf.cdf.quantiles, cf.cdf.probabilities, label=f"Median quantile: {median_quantile:.2f} and median: {median}")
plt.xlabel("data")
plt.xscale("log")
plt.ylabel("cdf")
plt.title("Empirical Sensitive CDF")
plt.legend()

# create folder in data/data_{N}
folder_name = f"data/N_{N}/B_exp_{B_exp}"
os.makedirs(f"{folder_name}", exist_ok=True)

# save figure
plt.savefig(f"{folder_name}/pareto_cdf.png")

# save data
with open(f"{folder_name}/pareto_data.pkl", "wb") as f:
    pickle.dump(data, f)

# save cdf
with open(f"{folder_name}/pareto_cdf.pkl", "wb") as f:
    pickle.dump(cf_dict, f)

# save median
with open(f"{folder_name}/pareto_median.pkl", "wb") as f:
    pickle.dump(median, f)

# save median quantile
with open(f"{folder_name}/pareto_median_quantile.pkl", "wb") as f:
    pickle.dump(median_quantile, f)

# save bins
with open(f"{folder_name}/pareto_bins.pkl", "wb") as f:
    pickle.dump(bins, f)

# save intervals
with open(f"{folder_name}/pareto_intervals.pkl", "wb") as f:
    pickle.dump(intervals, f)
