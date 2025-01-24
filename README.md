# Lightweight Protocols for Distributed Private Quantile Estimation

This repository contain the code for the paper "Lightweight Protocols for Distributed Private Quantile Estimation" 

To run the code, you need to install a conda environment using the environment.yml file
    
    conda env create -f environment.yml

Then, activate the environment

    conda activate LDP_q_est

## Mechanisms
*BaySS* contains the code for the differential private implementation of the Bayesian Screening Search in 

[Gretta, L. and Price, E. Sharp noisy binary search with
monotonic probabilities. In Bringmann, K., Grohe,
M., Puppis, G., and Svensson, O. (eds.), 51st Interna-
tional Colloquium on Automata, Languages, and Pro-
gramming, ICALP 2024, July 8-12, 2024, Tallinn, Es-
tonia, volume 297 of LIPIcs, pp. 75:1–75:19. Schloss
Dagstuhl - Leibniz-Zentrum f ¨ur Informatik, 2024. doi:
10.4230/LIPICS.ICALP.2024.75. URL https://doi.
org/10.4230/LIPIcs.ICALP.2024.75.]

More precisely, is an implementation of Algorithm 3 in which each coin flip is privatized using randomized response.

*hierarchical_mechanism* contains the code for the Hierarchical mechanism in 
[Kulkarni, Tejas. "Answering range queries under local differential privacy." Proceedings of the 2019 International Conference on Management of Data. 2019.].
The LDP protocol used is the unary encoding, we used the library from https://github.com/Samuel-Maddock/pure-LDP

*naive_noisy_binary_search* contains the code for a standard binary search mechanism with randomized response.

## Data Generation
Data is already provided in the experiments folders, however, you can generate new data by running the shell script in 
*run_commands/generate_data*. To generate different dataset it is sufficient to change the seed in the script.

## Experiments
In *Pareto_income_data* we provide the analysis for the hyper-parameter of the BaySS mechanism in **find_constant.py**, and 
also the comparison experiments on the other files. In *Skewed data* there are the experiments on a synthetic integer uniform distribution
with random left and right bounds. 
The experiments can be run by exectuing the shell scripts in *run_commands*.

