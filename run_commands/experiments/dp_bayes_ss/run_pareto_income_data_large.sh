# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd experiments/pareto_income_data || exit
# run the python program synthetic_dataset.py

# VERY COMPUTATIONALLY INTENSIVE
python experiment_DpBayeSS.py --N 10000000 --B_exp 8 --c 0.6