#!/bin/bash

models=(
    # Add your model identifiers here. For example:
    "euler/r1_sft_1.5b_sol_ckpts/71"
)

languages=(
    "Korean"
    "Afrikaans"
    "Arabic"
    "Chinese (Simplified)"
    "French"
    "English"
    "German"
    "Hebrew"
    "Indonesian"
    "Vietnamese"
    "Italian"
    "Japanese"
    "Spanish"
    "Turkish"
)

output_dir="outputs/"

# Convert the arrays into space-separated strings.
model_args="${models[@]}"
language_args="${languages[@]}"

# Run the evaluation once, passing all models and languages.
python src/eval.py --models $model_args \
                --languages $language_args \
                --output_path "$output_dir" \
                --dataset "amphora/m-aime-2024" \
                --max_model_len 8192



# python src/eval.py "$model" "$language" "$output_dir" "amphora/m-math500" 8192