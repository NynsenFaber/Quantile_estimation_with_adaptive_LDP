# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd experiments/pareto_income_data || exit
# run the python program synthetic_dataset.py

# delta 1e-8 and large N
python experiment_shuffle_DpNaiveNBS.py --N 10000000 --B_exp 8 --delta 1e-8  --num-exp 20