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
import matplotlib.pyplot as plt
import numpy as np
import pickle


def generate_pareto_data(shape, N, scale=2000):
    D = np.random.pareto(shape, N) * scale
    return D


N = 5000  # number of samples

data = generate_pareto_data(1.5, N)
cf = ecdf(data)
cf_dict = dict(zip(cf.cdf.quantiles, cf.cdf.probabilities))

# discretize the data
num_bins = int(1E5)  # Number of bins
max_value = 1E5
bins = np.linspace(0, int(max_value), num_bins + 1)  # Bin edges
intervals = np.array([bins[:-1], bins[1:]]).T

# HYPERPARAMETERS
num_experiments = 200
eps_list = np.linspace(0.1, 1, 5)
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


success_noisy = get_success_rate(noisy_binary_search)
print("Noisy Binary Search Completed", success_noisy)
success_bayes = get_success_rate(Gretta_Prica)
print("Bayesian Screening Search Completed")

# save the results in results folder
with open("results/pareto_success_noisy.pkl", "wb") as f:
    pickle.dump(success_noisy, f)
with open("results/pareto_success_bayes.pkl", "wb") as f:
    pickle.dump(success_bayes, f)

