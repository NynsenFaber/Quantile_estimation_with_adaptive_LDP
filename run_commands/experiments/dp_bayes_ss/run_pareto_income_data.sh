# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd experiments/pareto_income_data || exit
# run the python program synthetic_dataset.py

python experiment_DpBayeSS.py --N 2500 --B_exp 8 --c 0.6
python experiment_DpBayeSS.py --N 2500 --B_exp 9 --c 0.6
python experiment_DpBayeSS.py --N 5000 --B_exp 8 --c 0.6
python experiment_DpBayeSS.py --N 5000 --B_exp 9 --c 0.6
python experiment_DpBayeSS.py --N 7500 --B_exp 8 --c 0.6
python experiment_DpBayeSS.py --N 7500 --B_exp 9 --c 0.6