#!/bin/bash

# Defenses and datasets
defenses=("r2_nnm" "nnm_krum") # "ssmtd_hmm" "flame" "deepsight" "nnm" "autodfl"
datasets=("gtsrb") #   "gtsrb" "femnist"  "cifar" "fashionmnist"
attacks=("a3fl") # "omp" "neurotoxin"

# Loop over all combinations
for defense in "${defenses[@]}"; do
  for dataset in "${datasets[@]}"; do
    for attack in "${attacks[@]}"; do
      config="./src/experiment/configs/${defense}/${attack}_analysis_${dataset}.yml" 
      echo "Running experiment:"
      echo "Defense: $defense | Dataset: $dataset | Attack: $attack"
      echo "Config: $config"
      
      python main.py --config "$config"
      
      echo "---------------------------------------"
    done
  done
done

echo "All experiments finished."