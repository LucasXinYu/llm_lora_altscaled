#!/bin/bash


# chmod +x run_main_sft.sh
# bash run_main_sft.sh

optim=("sgdr" "sgd")
ranks=(8 16) #(8 16)
alt_opts=("True" "False")  # Renamed to avoid confusion

base_command="python main_sft.py"


for optim in "${optim[@]}"; do
    for rank in "${ranks[@]}"; do
        for alt_opt in "${alt_opts[@]}"; do
            command="$base_command --optim $optim --peft_lora_r $rank --alt_opt $alt_opt"
            echo "Running: $command"
            eval $command
        done
    done
done
