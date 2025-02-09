#!/bin/bash

# List of models to evaluate
models=(
    # Add your model identifiers here. For example:
    # "Qwen_Qwen2.5-1.5B-Instruct"
    # "Qwen_Qwen2.5-Math-1.5B-Instruct"
    # "nvidia_AceMath-1.5B-Instruct"
    # "deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B"
    # "ckpt/ab_ckpts/q25_1.5_50k_14"
    # "ckpt/sft_1.5b_214"
    # "gpt-4o-mini"
    # "gpt-4o"
    # "OLAIR/ko-r1-1.5b-preview"
    # "OLAIR/ko-r1-7b-preview"
    # "ckpts/ab_ckpts/q25_1.5_30k_14"
    # "ckpts/ab_ckpts/q25_1.5_4x6_24k"
    # "ckpts/ab_ckpts/q25_1.5_3x8_24k"
    # "ckpt/ab_ckpts/q25_1.5_50k_14"
    # "ckpts/lang-7"
    # "amphora/r1-grpo"
    # "Qwen/Qwen2.5-3B-Instruct"
    "amphora_3b-5k-en"
)

# List of languages to evaluate
# languages=("af" "ar" "bg" "bn" "ca" "cs" "cy" "da" "de" "el" "en" "es" "et" "fa" "fi" "fr" "gu" "he" "hi" "hr" "hu" "id" "it" "ja" "kn" "ko" "lt" "lv" "mk" "ml" "mr" "ne" "nl" "no" "pa" "pl" "pt" "ro" "ru" "sk" "sl" "so" "sq" "sv" "sw" "ta" "te" "th" "tl" "tr" "uk" "ur" "vi" "zh-cn" "zh-tw")
# languages=("bg" "bn" "ca" "cs" "cy" "da" "el" "et" "fa" "fi" "gu" "hi" "hr" "hu" "kn" "lt" "lv" "mk" "ml" "mr" "ne" "nl" "no" "pa" "pl" "pt" "ro" "ru" "sk" "sl" "so" "sq" "sv" "sw" "ta" "te" "th" "tl" "uk" "ur")
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
output_dir="lcs_new_results/"

datasets=(
    "OLAIR/mt-math-500"
    # "OLAIR/mt-math-extended"
    "OLAIR/mt-aime-extended"
    # "OLAIR/M-IMO-extended"
)

model_args="${models[@]}"

# Iterate over datasets
for dataset in "${datasets[@]}"; do
    echo "Running evaluation for model: $model on dataset: $dataset"
    python src/lcs.py --models $model_args \
                      --languages "${languages[@]}" \
                      --output_path "$output_dir" \
                      --dataset "$dataset"
done