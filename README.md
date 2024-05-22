# Quantile Estimation with Adaptive Local Differential Privacy

This repository contain the code for the paper "Quantile Estimation with Adaptive Local Differential Privacy" 

To run the code, you need to install a conda environment using the environment.yml file
    
    conda env create -f environment.yml

Then, activate the environment

    conda activate LDP_q_est

## Mechanisms
*BaySS* contains the code for the differential private implementation of the Bayesian Screening Search in 
[Gretta, Lucas, and Eric Price. "Sharp Noisy Binary Search with Monotonic Probabilities." arXiv preprint arXiv:2311.00840 (2023).]

*hierarchical_mechanism* contains the code for the Hierarchical mechanism in 
[Kulkarni, Tejas. "Answering range queries under local differential privacy." Proceedings of the 2019 International Conference on Management of Data. 2019.].
The LDP protocol used is the unary encoding, we used the library from https://github.com/Samuel-Maddock/pure-LDP

*naive_noisy_binary_search* contains the code for a standard binary search mechanism with randomized response.

## Experiments
In *Pareto_income_data* we provide the analysis for the hyper-parameter of the BaySS mechanism in **find_alpha.py**, and 
also the comparison experiments on the other files. In *Skewed data* there are the experiments on a synthetic integer uniform distribution
with random left and right bounds. 
The experiments python files contain the information of how to change the parameters (like the number of user, the privacy budget), while the
jupyter files contain the plots.
In each folder there is a **generate_data.py** used to generate and store the data.

