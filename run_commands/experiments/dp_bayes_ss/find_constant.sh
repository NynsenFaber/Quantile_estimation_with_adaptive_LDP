# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd Experiments/Pareto_income_data || exit
# run the python program synthetic_dataset.py

python find_constant.py --N 2500 --B_exp 9 --eps 0.5
python find_constant.py --N 2500 --B_exp 9 --eps 1.0
python find_constant.py --N 2500 --B_exp 9 --eps 1.5
python find_constant.py --N 5000 --B_exp 8 --eps 0.5
python find_constant.py --N 5000 --B_exp 8 --eps 1.0
python find_constant.py --N 5000 --B_exp 8 --eps 1.5