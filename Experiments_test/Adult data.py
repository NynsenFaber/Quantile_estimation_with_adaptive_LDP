# to get the parent directory
import os
import sys

# Get the current directory
current_dir = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from ucimlrepo import fetch_ucirepo
from scipy.stats import ecdf
from scipy.stats import bootstrap
from noisy_binary_search.mechanism import noisy_binary_search
from DP_GPNBS.mechanism import Gretta_Prica
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X = adult.data.features

data = X["age"].to_numpy()

# age from 0 to 100
B_max = 100
B = np.arange(0, B_max + 1)
intervals = np.array([B[:-1], B[1:]]).T

# get the empirical cdf of the coins
cf = ecdf(data)
cf_dict = dict(zip(cf.cdf.quantiles, cf.cdf.probabilities))

# HYPERPARAMETERS
eps_list = [0.1, 0.5, 1]
success_prob = 0.85
num_experiments = 100
alpha_list = np.linspace(0.4, 0.0001, 500)


def get_error(data, alpha=0.9):
    """
    Get the error (left, right) of the data using bootstrap
    """
    confidence_interval = []
    mean = np.mean(data)
    if np.std(data) > 0:
        res = bootstrap((data,), np.mean, confidence_level=alpha)
        confidence_interval.append(res.confidence_interval)
    else:
        confidence_interval.append((mean, mean))
    error = np.array(confidence_interval).flatten()
    return error


def get_alpha_bar(mechanism):
    # Hyperparameters
    M = len(data)
    replacement = False
    alpha_intervals = []
    start = 0
    for eps in eps_list:
        success = np.zeros(num_experiments)
        flag_low_error = True
        for a, alpha in enumerate(alpha_list[start:]):
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
            low_error, high_error = get_error(success, 0.85)
            if low_error < success_prob and flag_low_error:
                flag_low_error = False
                print(f'eps {eps} right interval alpha {alpha}')
                right_alpha = alpha
                start = max(a - 10, 0)
            if high_error < success_prob:
                print(f'eps {eps} left interval alpha {alpha}')
                left_alpha = alpha
                alpha_intervals.append((left_alpha, right_alpha))
                break
    return alpha_intervals


alpha_noisy = get_alpha_bar(noisy_binary_search)
print("Noisy Binary Search", alpha_noisy)
alpha_bayes = get_alpha_bar(Gretta_Prica())
print("Bayesian Screening Search", alpha_bayes)


def plot(alpha_noisy, alpha_bayes, eps_list):
    eps_list = np.array(eps_list)
    # compute the midle point of the error bars
    mid_noisy = [(noisy[0] + noisy[1]) / 2 for noisy in alpha_noisy]
    mid_bayes = [(bayes[0] + bayes[1]) / 2 for bayes in alpha_bayes]

    # Extract lower and upper error bar values for alpha_noisy and alpha_bayes
    noisy_lower = [mid_noisy[i] - noisy[0] for i, noisy in enumerate(alpha_noisy)]
    noisy_upper = [noisy[1] - mid_noisy[i] for i, noisy in enumerate(alpha_noisy)]

    bayes_lower = [mid_bayes[i] - bayes[0] for i, bayes in enumerate(alpha_bayes)]
    bayes_upper = [bayes[1] - mid_bayes[i] for i, bayes in enumerate(alpha_bayes)]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.errorbar(eps_list, mid_noisy, yerr=[noisy_lower, noisy_upper], fmt='none', ecolor='blue', elinewidth=2,
                 capsize=8, capthick=3, label='Noisy Binary Search')
    plt.errorbar(eps_list + 0.005, mid_bayes, yerr=[bayes_lower, bayes_upper], fmt='none', ecolor='red', elinewidth=2,
                 capsize=8, capthick=3, label='Bayesian Screening Search')

    plt.xlabel('Eps List', fontsize=12)
    plt.ylabel('Error Bars', fontsize=12)
    plt.title('Error vs Privacy budget with 85% success probability')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()
    plt.savefig('error_vs_eps.png')
    plt.show()


plot(alpha_noisy, alpha_bayes, eps_list)
