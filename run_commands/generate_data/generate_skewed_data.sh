# go back two folder
cd ../..
# enter in the folder preprocess_data
cd experiments/skewed_data || exit
# run the python program synthetic_dataset.py

python generate_data.py --seed 42 --N 2500 --B 100
python generate_data.py --seed 42 --N 2500 --B 1000
python generate_data.py --seed 42 --N 2500 --B 10000
python generate_data.py --seed 42 --N 2500 --B 100000
python generate_data.py --seed 42 --N 2500 --B 1000000
python generate_data.py --seed 42 --N 2500 --B 10000000