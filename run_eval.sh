#!/bin/bash

models=(
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "Qwen/Qwen2.5-Math-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    # "nvidia/AceMath-7B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "CohereForAI/aya-expanse-8b"
    # "CohereForAI/c4ai-command-r7b-12-2024"
    "google/gemma-2-9b-it"
    # "meta-llama/Llama-3.2-1B-Instruct"
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
# Loop through models and languages
for model in "${models[@]}"; do
    for language in "${languages[@]}"; do
        echo "Running model: $model for language: $language"
        python eval.py "$model" "$language" "$output_dir" "amphora/m-math500"
    done
done

# # Define models and languages
# models=(
#     "Qwen/Qwen2.5-1.5B-Instruct"
#     "nvidia/AceMath-1.5B-Instruct"
#     "Qwen/Qwen2.5-Math-1.5B-Instruct"
#     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#     "OLAIR/test-models"
#     # "meta-llama/Llama-3.2-1B-Instruct"
# )

# # Loop through models and languages
# for model in "${models[@]}"; do
#     for language in "${languages[@]}"; do
#         echo "Running model: $model for language: $language"
#         python eval.py "$model" "$language" "$output_dir" "OLAIR/M-IMO"
#     done
# done

# echo "All tasks completed."