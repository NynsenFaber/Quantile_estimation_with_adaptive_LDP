# to get the parent directory
import os
import sys

# Get the current directory
current_dir = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from scipy.stats import ecdf
from scipy.stats import bootstrap
import tqdm
from noisy_binary_search.mechanism import noisy_binary_search
from DP_GPNBS.mechanism import Gretta_Prica

from b_ary_mechanism.hierarchical_mechanism import hierarchical_mechanism_quantile, get_intervals
from b_ary_mechanism.data_structure import Tree

import numpy as np
import pickle


def generate_pareto_data(shape, N, scale=2000):
    D = np.random.pareto(shape, N) * scale
    return D


N = 5000  # number of samples

# generate the data
data = generate_pareto_data(1.5, N)

# discretize the data
num_bins = int(4 ** 9)  # Number of bins
bins = np.array(range(num_bins))  # Bin edges
intervals = np.array([bins[:-1], bins[1:]]).T
data = np.round(data).astype(int)  # integer data
data = np.clip(data, 0, num_bins - 1)

# get the empirical cdf of the coins
cf = ecdf(data)
cf_dict = dict(zip(cf.cdf.quantiles, cf.cdf.probabilities))

for i, j in enumerate(cf_dict.keys()):
    if cf_dict[j] > 0.5:
        median = list(cf_dict.keys())[i-1]
        break



# HYPERPARAMETERS
num_experiments = 200
eps_list = np.linspace(1.1, 10, 5)
alpha = 0.05


def get_error(data, cl=0.9):
    """
    Get the error (left, right) of the data using bootstrap
    """
    confidence_interval = []
    mean = np.mean(data)
    if np.std(data) > 0:
        res = bootstrap((data,), np.mean, confidence_level=cl)
        confidence_interval.append(res.confidence_interval)
    else:
        confidence_interval.append((mean, mean))
    error = np.array(confidence_interval).flatten()
    low_error = mean - error[0]
    high_error = error[1] - mean
    return mean, low_error, high_error


# --------------- start with hierarchical mechanism ----------------#
print("Hierarchical Mechanism")
tree = Tree(bins, branching=4)
intervals_hh = get_intervals(tree, branching=4)
success_rate_hh = np.zeros((len(eps_list), 3))
for e, eps in tqdm.tqdm(enumerate(eps_list)):
    success = np.zeros(num_experiments)
    for i in range(num_experiments):
        D = list(data)
        output = hierarchical_mechanism_quantile(tree=tree,
                                                 data=D,
                                                 protocol='unary_encoding',
                                                 eps=eps,
                                                 quantile=0.5,
                                                 intervals=intervals_hh,
                                                 replacement=False)
        # check if output is correct
        if output in cf_dict.keys():
            quantile_output = cf_dict[output]
        else:
            # find the higher value in the dictionary smaller than output
            if output >= min(cf_dict.keys()):
                quantile_output = cf_dict[max([x for x in cf_dict.keys() if x <= output])]
            else:
                quantile_output = 0
        if 0.5 - alpha < quantile_output < 0.5 + alpha:
            success[i] = 1
        else:
            success[i] = 0

    mean, low_error, high_error = get_error(success, 0.85)
    success_rate_hh[e] = np.array([mean, low_error, high_error])

print("Hierarchical Mechanism Completed", success_rate_hh)


def get_success_rate(mechanism):
    # Hyperparameters
    M = len(data)
    replacement = False
    success_rate = np.zeros((len(eps_list), 3))
    for e, eps in tqdm.tqdm(enumerate(eps_list)):
        success = np.zeros(num_experiments)
        for i in range(num_experiments):
            D = list(data)
            output = mechanism(D=D, intervals=intervals, alpha=alpha, eps=eps, M=M, replacement=replacement)
            # check if output is correct
            if output in cf_dict.keys():
                quantile_output = cf_dict[output]
            else:
                # find the higher value in the dictionary smaller than output
                if output >= min(cf_dict.keys()):
                    quantile_output = cf_dict[max([x for x in cf_dict.keys() if x <= output])]
                else:
                    quantile_output = 0
            if 0.5 - alpha < quantile_output < 0.5 + alpha:
                success[i] = 1
            else:
                success[i] = 0

        mean, low_error, high_error = get_error(success, 0.85)
        success_rate[e] = np.array([mean, low_error, high_error])

    return success_rate


print("\nNoisy Binary Search")
success_noisy = get_success_rate(noisy_binary_search)
print("Noisy Binary Search Completed", success_noisy)
print("\nBayesian Screening Search")
success_bayes = get_success_rate(Gretta_Prica)
print("Bayesian Screening Search Completed", success_bayes)

# save the results in results folder
with open("results_high_privacy/pareto_success_hh.pkl", "wb") as f:
    pickle.dump(success_rate_hh, f)
with open("results_high_privacy/pareto_success_noisy.pkl", "wb") as f:
    pickle.dump(success_noisy, f)
with open("results_high_privacy/pareto_success_bayes.pkl", "wb") as f:
    pickle.dump(success_bayes, f)
