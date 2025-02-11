#!/bin/bash

# List of models to evaluate
models=(
    # Add your model identifiers here. For example:
    "jwhj/Qwen2.5-Math-1.5B-OREO"
    "nvidia/AceMath-1.5B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-1.5B-Instruct"
    "PRIME-RL/Eurus-2-7B-PRIME"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "nvidia/AceMath-7B-Instruct"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "o3-mini"
)

# List of languages to evaluate
languages=("Afrikaans" "Albanian" "Arabic" "Bengali" "Bulgarian" "Catalan" "Chinese (Simplified)" "Chinese (Traditional)" "Croatian" "Czech" "Danish" "Dutch" "English" "Estonian" "Finnish" "French" "German" "Greek" "Gujarati" "Hebrew" "Hindi" "Hungarian" "Indonesian" "Italian" "Japanese" "Kannada" "Korean" "Latvian" "Lithuanian" "Macedonian" "Malayalam" "Marathi" "Nepali" "Norwegian" "Persian" "Polish" "Portuguese" "Punjabi" "Romanian" "Russian" "Slovak" "Slovenian" "Somali" "Spanish" "Swahili" "Swedish" "Tagalog" "Tamil" "Telugu" "Thai" "Turkish" "Ukrainian" "Urdu" "Vietnamese" "Welsh")

# Output directory
output_dir="lcs_results/"

datasets=(
    "OLAIR/mt-math-500"
    "OLAIR/mt-math-extended"
    "OLAIR/mt-aime-extended"
    "OLAIR/M-IMO-extended"
    "OLAIR/MMO"
)

python src/lcs.py --models "${models[@]}" \
                  --datasets "${datasets[@]}" \
                  --languages "${languages[@]}" \
                  --output_path $output_dir