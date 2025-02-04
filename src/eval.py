import argparse
import os
import json
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import huggingface_hub

# Log in to Hugging Face Hub (make sure your token is correct)
huggingface_hub.login("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")

def evaluate_model_for_language(llm, tokenizer, df, language, sampling_params, model_path, output_path, dataset):
    # Determine the column name for the current language.
    # For English, assume the original column "problem" is used.
    col_name = f"problem_{language}"
    if language.lower() in ["english", "en"]:
        col_name = "problem"
    
    if col_name not in df.columns:
        print(f"Error: Language column '{col_name}' not found in dataset for language '{language}'. Skipping...")
        return

    # Prepare prompts for each question
    qrys = []
    for p in df[col_name].values:
        if p is None:
            message = [
                {'role': 'system', 'content': 'Return your response in \\boxed{} format.'},
                {'role': 'user', 'content': ""},
            ]
        else:
            message = [
                {'role': 'system', 'content': 'Return your response in \\boxed{} format.'},
                {'role': 'user', 'content': p},
            ]
        # Create text from the chat template.
        text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        qrys.append(text)

    # Generate responses for all questions
    outputs = llm.generate(qrys, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    # Prepare output directory and filename.
    model_name = model_path.replace('/', '_')
    dataset_name = dataset.replace('/', '_')
    # Replace spaces in language with underscores for the filename.
    lang_clean = language.replace(" ", "_")
    out_dir = os.path.join(output_path, dataset_name, model_name)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{lang_clean}.jsonl")

    # Save each record as a JSONL line.
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            record = {
                "original_question": df["problem"].iloc[i],
                "question": df[col_name].iloc[i],
                "response": responses[i] if i < len(responses) else "",
                "answer": df["answer"].iloc[i]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results for language '{language}' saved to {output_file}")

def main(models, languages, output_path, dataset, max_model_len):
    # Set sampling parameters (note: ensure max_tokens is an integer)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=int(max_model_len * 0.8))
    
    # Load dataset (assuming the split "train" exists)
    ds = load_dataset(dataset)
    df = pd.DataFrame(ds['train'])
    if len(df) > 100:
        df = df.sample(1)  # Sample 1 example if dataset is large
    
    # Loop through each model.
    for model_path in models:
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = LLM(model=model_path, max_model_len=max_model_len, tensor_parallel_size=torch.cuda.device_count())
        
        # For each model, evaluate all requested languages.
        for language in languages:
            print(f"Running model: {model_path} for language: {language}")
            evaluate_model_for_language(llm, tokenizer, df, language, sampling_params, model_path, output_path, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLLM model on math problems for specified languages."
    )
    parser.add_argument("--models", nargs="+", required=True, help="List of model paths.")
    parser.add_argument("--languages", nargs="+", required=True, help="List of languages (e.g., Korean, Chinese, etc.).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL files.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset identifier (e.g., 'amphora/m-aime-2024').")
    parser.add_argument("--max_model_len", type=int, required=True, help="Maximum token length for the model.")
    args = parser.parse_args()
    
    main(args.models, args.languages, args.output_path, args.dataset, args.max_model_len)
