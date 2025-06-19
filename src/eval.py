import argparse
import os
import json
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from litellm import batch_completion
from prompts import lang_dict, language_dict, dataset_name_dict, system_prompt_dict
from openai import OpenAI
import yaml
import re
from dotenv import load_dotenv
from prompt import *
from tqdm import tqdm

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
hf_token = os.getenv('HF_TOKEN')

sft_dict = {"English": "english", "Chinese_(Simplified)": "chinese", "Bulgarian": "bulgarian", "Swahili": "swahili", "Somali": "somali", "Japanese": "japanese", "French": "french", "Latvian": "latvian"}
boxed_dict = {"English": "Return your final response within \\boxed{{}}.", "Chinese_(Simplified)": "请在\\boxed{{}}内返回你的最终回答。"}

def get_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value][0]

def lang_detect(l):
    if l in language_dict.keys():
        return l
    else:
        return get_keys_by_value(language_dict, l)

def dataset_language_detect(dataset, lang_type):
    if not lang_type:
        if dataset in ["math100", "aime2024", "IMO"]:
            return lang_dict[55]
        elif dataset in ["MMO"]:
            return lang_dict["MMO"]
        else:
            raise TypeError(f"Sorry! {dataset} does not suppoorted yet.")
    elif type(lang_type) == list:
        return lang_type
    else:
        return lang_dict[lang_type]

def system_prompt_select(model, system_prompt_dict=system_prompt_dict):
    return system_prompt_dict[model] if model in list(system_prompt_dict.keys()) else 'Return your response in \\boxed{} format.'

def message_generate(model, query, tokenizer, system_lang, prefix=None):
    if prefix:
        message = [
            {'role': 'system', 'content': prefix},
            {'role': 'user', 'content': query if query else ""},
        ]
    else:
        message = [
            {'role': 'system', 'content': system_prompt_select(sft_dict[system_lang])},
            {'role': 'user', 'content': query if query else ""},
        ]
    return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

def evaluate_model_for_language(llm, tokenizer, df, language, system_lang, sampling_params, model_path, output_path, dataset, language_forcing):
    # Determine the column name for the current language.
    # For English, assume the original column "problem" is used.
    col_name = lang_detect(language) if "MMO" not in dataset else "question"
        
    if col_name not in df.columns:
        print(f"Error: Language column '{col_name}' not found in dataset for language '{language}'. Skipping...")
        return

    if isinstance(llm, str):
        qrys = []
        for p in df[col_name].values:
            qrys.append([
                {"role": "system", "content": "Return your response in \\boxed{} format."},
                {"role": "user", "content": p}
            ])
        outputs = batch_completion(
            model=llm,
            messages=qrys,
            temperature=0,
            max_tokens=int(8192),
            stop=['<|end_of_solution|>']
        )
        responses = [output.choices[0].message.content for output in outputs]
        print("Completion Succeed!")
    else:
        # Prepare prompts for each question
        questions = df[col_name].values
        if language_forcing:
            force_prefix = distill_prefix_template.format(language=system_lang)
            prompt = translate_template.format(source_text=force_prefix, language=system_lang)
            translated_prefix = generate_response_with_retries([prompt], "gemini-2.5-flash-preview-05-20", "<translation>", "</translation>")[0]
            qrys = [message_generate(model_path, p, tokenizer, system_lang, translated_prefix) for p in df[col_name].values]
        else:
            qrys = [message_generate(model_path, p, tokenizer, system_lang) for p in df[col_name].values]
        print(qrys[0])
        # Generate responses for all questions
        outputs = llm.generate(qrys, sampling_params)
        
        # Process multiple samples per question
        responses = []
        for output in outputs:
            question_responses = []
            for candidate in output.outputs:
                question_responses.append(candidate.text)
            responses.append(question_responses)

        correct = []
        all_responses_processed = []

        for i, question_responses in tqdm(enumerate(responses), total=len(responses)):
            question_correct = []
            for response in question_responses:
                try:
                    answer = response.split("<|begin_of_solution|>")[1].strip()
                except:
                    question_correct.append(0)
                    all_responses_processed.append(response)
                    continue
                #try:
                #    prompt = extract_answer_template.format(answer=answer)
                #    answer = generate_response_with_retries([prompt], "gpt-4.1-mini", "```", "```")[0]
                #except:
                #    question_correct.append(0)
                #    all_responses_processed.append(response)
                #    continue
                
                if answer:
                    gold = str(df["answer"].iloc[i])
                    prompt = verify_answer_template.format(problem=questions[i], gt_solution=gold, model_solution=answer)
                    matches = generate_response_with_retries([prompt], "gpt-4.1-mini", "```", "```")[0]
                    if matches is None:
                        question_correct.append(0)
                    elif "True" in matches:
                        question_correct.append(1)
                    else:
                        question_correct.append(0)
                else:
                    question_correct.append(0)
                all_responses_processed.append(response)
            
            correct.append(question_correct)
        
    # Prepare output directory and filename.
    model_name = model_path.replace('/', '_')
    # Replace spaces in language with underscores for the filename.
    lang_clean = language.replace(" ", "_")
    out_dir = os.path.join(output_path, dataset, model_name)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{lang_clean}.jsonl")

    # Save each record as a JSONL line.
    if "level" and "subject" in df.columns:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(df)):
                for sample_idx in range(len(responses[i])):
                    record = {
                        "original_question": df[col_name].iloc[i],
                        "question": df[col_name].iloc[i],
                        "response": str(responses[i][sample_idx]),
                        "answer": str(df["answer"].iloc[i]),
                        "correct": str(correct[i][sample_idx]),
                        "level": str(df["level"].iloc[i]),
                        "subject": str(df["subject"].iloc[i]),
                        "sample_idx": sample_idx
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")    
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(len(df)):
                for sample_idx in range(len(responses[i])):
                    record = {
                        "original_question": df[col_name].iloc[i],
                        "question": df[col_name].iloc[i],
                        "response": str(responses[i][sample_idx]),
                        "answer": str(df["answer"].iloc[i]),
                        "correct": str(correct[i][sample_idx]),
                        "sample_idx": sample_idx
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results for language '{language}' saved to {output_file}")

def main(models, datasets, lang_type, system_lang, sample, output_path, max_model_len=16384, n_samples=1, language_forcing=False):

    # Set sampling parameters (note: ensure max_tokens is an integer)
    client = OpenAI()
    openai_models = [m.id for m in client.models.list().data]
    sampling_params = SamplingParams(n=n_samples, temperature=0.6, top_p=0.95, 
                                   max_tokens=int(max_model_len), stop=['<|end_of_solution|>'])

    for model in models:
        revision = model.split("***")[-1] if "***" in model else "main"
        model_name = model.split("***")[0].replace("/", "_")
        print(f"Loading model: {model_name} - {revision}")
        if model not in openai_models:
            tokenizer = AutoTokenizer.from_pretrained(model)
            llm = LLM(model=model, max_model_len=max_model_len, 
                      tensor_parallel_size=torch.cuda.device_count(), revision=revision)
        else:
            llm, tokenizer = model, None

        for data in datasets:
            # Load dataset (assuming the split "train" exists)
            if "MMO" in data:
                ds = load_dataset("OLAIR/MMO")
            elif "math500" in data:
                df = load_from_disk(f"/scratch/current/joshbarua/datasets/mt-math-500").to_pandas()
            elif "math7500" in data:
                df = load_from_disk(f"/scratch/current/joshbarua/datasets/mt-math-7500").to_pandas()
            elif "gpqa" in data:
                df = load_from_disk(f"/scratch/current/joshbarua/datasets/benchmax_gpqa").to_pandas()
            else:
                df = load_dataset("amphora/MCLM", dataset_name_dict[data], split="test").to_pandas()
            if sample:
                df = df.sample(10,random_state=1210)  # Sample 10 example if dataset is large

            # For each model, evaluate all requested languages.
            for language in dataset_language_detect(data, lang_type):
                if "MMO" in data:
                    df = ds[language].to_pandas()
                if os.path.exists(os.path.join(output_path, data, model.replace("/", "_"), f"{language}.jsonl")):
                    continue
                print(f"Running model: {model_name} - {revision} for language: {language}")
                evaluate_model_for_language(llm, tokenizer, df, language, system_lang, sampling_params, model.split("***")[0], output_path, data, language_forcing)
        del llm
        torch.cuda.empty_cache()

if __name__ == "__main__":
    with open("eval.yaml", "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    os.environ["HF_TOKEN"] = hf_token
    os.environ["OPENAI_API_KEY"] = openai_api_key

    if not args["system_language"]:
        args["system_language"] = args["language_type"]

    if not args["language_forcing"]:
        args["language_forcing"] = False
    
    main(args["models"], args["datasets"], args["language_type"], args["system_language"], args["samples"], args["output_path"], args["max_model_len"], args["n_samples"], args["language_forcing"])