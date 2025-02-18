export HF_TOKEN="hf_xxx" # Your Huggingface Token
export OPENAI_API_KEY="sk-xxx" # Your OpenAI API KEY

python src/mo_score_litellm.py \
    --path_list \
    --models \
    --datasets "IMO" "MMO" \
    --languages \
    --judge "gpt-4o-mini"