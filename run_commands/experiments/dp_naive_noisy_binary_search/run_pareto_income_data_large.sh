# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd experiments/pareto_income_data || exit

python experiment_DpNaiveNBS.py --N 10000000 --B_exp 8  --num-exp 20