import argparse
import os
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import huggingface_hub
import torch

huggingface_hub.login("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")

def main(model_path, language, output_path, dataset):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    
    # Load dataset
    df = load_dataset(dataset)[language].to_pandas()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path,max_model_len=4096, tensor_parallel_size=torch.cuda.device_count())
    
    qrys = []
    for p in df["question"].values:
        message = [
            {'role': 'system', 'content': 'Return your response in \\boxed{} format.'}, 
            {'role': 'user', 'content': p}, 
        ]
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        qrys.append(text)
    
    # Generate responses
    outputs = llm.generate(qrys, sampling_params)
    response = [output.outputs[0].text for output in outputs]
    
    # Prepare output directory and filename
    model_name = model_path.replace('/', '_')
    dataset_name = dataset.replace('/', '_')
    output_dir = os.path.join(output_path, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{language}.jsonl")
    
    # Save results to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            record = {
                "original_question": df["question"].iloc[i],
                "question": df["question"].iloc[i],
                "response": response[i] if i < len(response) else "",
                "answer": df["answer"].iloc[i]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLLM model on math problems in a specified language.")
    parser.add_argument("model_path", type=str, help="Path to the model.")
    parser.add_argument("language", type=str, help="Language for the problems (e.g., Korean, Chinese, etc.).")
    parser.add_argument("output_path", type=str, help="Path to save output JSONL file.")
    parser.add_argument("dataset", type=str, help="Path to dataset")
    args = parser.parse_args()
    
    main(args.model_path, args.language, args.output_path, args.dataset)
