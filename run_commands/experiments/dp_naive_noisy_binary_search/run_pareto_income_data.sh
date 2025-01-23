# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd experiments/pareto_income_data || exit
# run the python program synthetic_dataset.py

python experiment_DpNaiveNBS.py --N 2500 --B_exp 8
python experiment_DpNaiveNBS.py --N 2500 --B_exp 9
python experiment_DpNaiveNBS.py --N 5000 --B_exp 8
python experiment_DpNaiveNBS.py --N 5000 --B_exp 9
python experiment_DpNaiveNBS.py --N 7500 --B_exp 8
python experiment_DpNaiveNBS.py --N 7500 --B_exp 9