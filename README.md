# Quantile Estimation with Adaptive Local Differential Privacy

This repository contain the code for the paper "Quantile Estimation with Adaptive Local Differential Privacy" 

To run the code, you need to install a conda environment using the environment.yml file
    
    ``` conda env create -f environment.yml ```

Then, activate the environment

    ``` conda activate LDP_q_est```

## Mechanisms
*BaySS* contain the code for the differential private implementation of the Bayesian Screening Search in 
[Gretta, Lucas, and Eric Price. "Sharp Noisy Binary Search with Monotonic Probabilities." arXiv preprint arXiv:2311.00840 (2023).]

*hierarchical_mechanism* contain the code for the Hierarchical mechanism in 
[Kulkarni, Tejas. "Answering range queries under local differential privacy." Proceedings of the 2019 International Conference on Management of Data. 2019.]
