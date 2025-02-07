import argparse
import os
import json
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import huggingface_hub
from prompts import system_prompt

# Log in to Hugging Face Hub (make sure your token is correct)
huggingface_hub.login("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")

language_dict = {
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'ar': 'Arabic',
    'bn': 'Bengali',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'kn': 'Kannada',
    'ko': 'Korean',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'pa': 'Punjabi',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'es': 'Spanish',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'tl': 'Tagalog',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'cy': 'Welsh'
}

def get_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value]

def evaluate_model_for_language(llm, tokenizer, df, language, sampling_params, model_path, output_path, dataset):
    # Determine the column name for the current language.
    # For English, assume the original column "problem" is used.
    if len(language) == 2:
        col_name = language
    else:
        col_name = get_keys_by_value(language_dict, language)[0]
        
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
            if "euler" in model_path:
                message = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': p},
                    {'role': 'assistant', 'content': '<think>'},
                ]
            else:
                message = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': p}
                ]
        if "euler" in model_path: 
            # Create text from the chat template.
            text = tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
        else:
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
                "original_question": df[col_name].iloc[i],
                "question": df[col_name].iloc[i],
                "response": str(responses[i]) if i < len(responses) else "",
                "answer": str(df["answer"].iloc[i])
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results for language '{language}' saved to {output_file}")

def main(model_path, languages, output_path, dataset, max_model_len, sample):
    # Set sampling parameters (note: ensure max_tokens is an integer)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=int(max_model_len * 0.8))
    
    # Load dataset (assuming the split "train" exists)
    ds = load_dataset(dataset)
    df = pd.DataFrame(ds['train'])
    if sample=='True':
        df = df.sample(100,random_state=1210)  # Sample 100 example if dataset is large

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
    parser.add_argument("--model_path", type=str, required=True, help="Model path.")
    parser.add_argument("--languages", nargs="+", required=True, help="List of languages (e.g., Korean, Chinese, etc.).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL files.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset identifier (e.g., 'amphora/m-aime-2024').")
    parser.add_argument("--max_model_len", type=int, required=True, help="Maximum token length for the model.")
    parser.add_argument("--sample", type=bool, required=True, help="Sample or not.")
    
    args = parser.parse_args()
    
    main(args.model_path, args.languages, args.output_path, args.dataset, args.max_model_len, args.sample)
