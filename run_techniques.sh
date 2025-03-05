#!/bin/bash
# Array of datasets
datasets=("adult" "compas" "german")

# Array of models
models=("logistic_regression" "random_forest")
techniques=("reweighing" "laplace_mechanism")
# Number of repetitions
repetitions=10

# Directory to save results
results_dir="results"
mkdir -p $results_dir

# Path to the virtual environment activation script
venv_path="C:/Users/iolem/Desktop/Tesi/venv/Scripts/activate"

# Activate the virtual environment
source $venv_path

# Loop through all combinations

  # Loop through all combinations
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for technique in "${techniques[@]}"; do
      for ((i=1; i<=repetitions; i++)); do
        echo "Running: dataset=$dataset, model=$model, technique=$technique, repetition=$i"
        output_file="${results_dir}/${dataset}_${model}_${technique}_run${i}.txt"
              python main.py --dataset $dataset --model $model --technique $technique >> $output_file 2>/dev/null
      done
    done
  done
done

# Deactivate the virtual environment
deactivate
echo "Finished all repetitions. Results saved in $output_file."


