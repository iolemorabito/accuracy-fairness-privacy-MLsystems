#!/bin/bash

# Array of datasets
datasets=("adult")

# Array of models
models=("logistic_regression" "random_forest")
techniques=("anonymization")
# Array of accuracy metrics
accuracy_metrics=("balanced_accuracy" "precision" "recall" "f1_score")

# Array of fairness metrics
fairness_metrics=("statistical_parity_difference" "equal_opportunity_difference" "average_odds_difference")

# Array of privacy metrics
privacy_metrics=("pdtp" "SHAPr" "differential_privacy_loss")

# Number of repetitions
repetitions=10

# Directory to save results
results_dir="results_techniques"
mkdir -p $results_dir

# Path to the virtual environment activation script
venv_path="C:/Users/iolem/Desktop/Tesi/venv/Scripts/activate"

# Activate the virtual environment
source $venv_path

# Loop through all combinations
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for technique in "${techniques[@]}"; do
      for ((i=1; i<=repetitions; i++)); do
        echo "Running: dataset=$dataset,model=$model,technique=$technique, repetition=$i"
        output_file="${results_dir}/${dataset}_${model}_${technique}_run${i}.txt"
              python anon.py --dataset $dataset --model $model --technique $technique  >> $output_file 2>/dev/null
      done
    done
  done
done

# Deactivate the virtual environment
deactivate
