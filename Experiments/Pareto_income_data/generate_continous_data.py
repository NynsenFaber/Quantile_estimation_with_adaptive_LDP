import numpy as np
from scipy.stats import ecdf
import matplotlib.pyplot as plt
import pickle
import os


def generate_pareto_data(shape, N):
    return np.random.pareto(shape, N) * 2000


seed = 42
N = 20  # number of samples

np.random.seed(seed)

# generate the data
data = generate_pareto_data(1.5, N)

# get the empirical cdf of the coins
cf = ecdf(data)
cf_dict = dict(zip(cf.cdf.quantiles, cf.cdf.probabilities))
cf_dict[0] = 0
# sort by key
cf_dict = dict(sorted(cf_dict.items()))

# get the median quantile
median = None
for i, j in enumerate(cf_dict.keys()):
    if cf_dict[j] > 0.5:
        median = list(cf_dict.keys())[i - 1]
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
folder_name = f"data/continuum/N_{N}"
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
