# go back two folder
cd ../../..
# enter in the folder preprocess_data
cd Experiments/skewed_data || exit
# run the python program synthetic_dataset.py

python experiment_DpNaiveNBS.py --N 2500 --eps 0.5
python experiment_DpNaiveNBS.py --N 2500 --eps 1