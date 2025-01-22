# go back two folder
cd ../..
# enter in the folder preprocess_data
cd Experiments/Pareto_income_data || exit
# run the python program synthetic_dataset.py

python generate_data.py --seed 42 --N 2500 --B_exp 8
python generate_data.py --seed 42 --N 2500 --B_exp 9
python generate_data.py --seed 42 --N 5000 --B_exp 8
python generate_data.py --seed 42 --N 5000 --B_exp 9
python generate_data.py --seed 42 --N 7500 --B_exp 8
python generate_data.py --seed 42 --N 7500 --B_exp 9