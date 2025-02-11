#!/bin/bash

# List of models to evaluate
models=(
    # Add your model identifiers here. For example:
    # "ckpts/ab_ckpts/q25_1.5_50k_14"
    # "OLAIR/ko-r1-1.5b-preview"
    # "OLAIR/ko-r1-7b-preview"
    # "ckpt/ab_ckpts/q25_1.5_30k_14"
    # "ckpt/ab_ckpts/q25_1.5_4x6_24k"
    # "ckpt/ab_ckpts/q25_1.5_4x28_112k"
    # "ckpt/ab_ckpts/q25_1.5_50k_14"
    # "amphora/r1-grpo"
    # "ckpt/sft_1.5b/214"
    # "ckpt/ab_ckpts/q25_1.5_50k_14"
)

# List of languages to evaluate
# languages=("af" "ar" "bg" "bn" "ca" "cs" "cy" "da" "de" "el" "en" "es" "et" "fa" "fi" "fr" "gu" "he" "hi" "hr" "hu" "id" "it" "ja" "kn" "ko" "lt" "lv" "mk" "ml" "mr" "ne" "nl" "no" "pa" "pl" "pt" "ro" "ru" "sk" "sl" "so" "sq" "sv" "sw" "ta" "te" "th" "tl" "tr" "uk" "ur" "vi" "zh-cn" "zh-tw")
languages=(
    "Turkish"
    "Spanish"
    "Japanese"
    "Italian"
    "Vietnamese"
    "Indonesian"
    "Hebrew"
    "German"
    "English"
    "French"
    "Chinese (Simplified)"
    "Arabic"
    "Afrikaans"
    "Korean"
)

# Output directory
output_dir="results/"

datasets=(
    "OLAIR/mt-math-500"
    # "OLAIR/mt-math-extended"
    "OLAIR/mt-aime-extended"
    "OLAIR/M-IMO-extended"
)

# Iterate over models
for model in "${models[@]}"; do    
    # Iterate over datasets
    for dataset in "${datasets[@]}"; do
        echo "Running evaluation for model: $model on dataset: $dataset"
        python src/eval.py --model_path "$model" \
                           --languages "${languages[@]}" \
                           --output_path "$output_dir" \
                           --dataset "$dataset" \
                           --max_model_len 8192 \
                           --sample "False"
    done
done
