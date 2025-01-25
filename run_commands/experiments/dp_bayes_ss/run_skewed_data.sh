# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd experiments/skewed_data || exit
# run the python program synthetic_dataset.py

python experiment_DpBayeSS.py --N 2500 --eps 0.5 --c 0.6
python experiment_DpBayeSS.py --N 2500 --eps 1.0 --c 0.6