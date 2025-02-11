export OPENAI_API_KEY="sk-xxx" # Your OpenAI API KEY

python src/mo_score.py \
    --models \
    --datasets \
    --score_type "data_collect" # ["data_collect", "send_batch", "receive_batch", "score"]