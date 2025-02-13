import argparse
import os
import json
import pandas as pd
import torch
from datasets import load_dataset
import huggingface_hub
from litellm import batch_completion
from prompts import lang_dict, language_dict, dataset_name_dict, system_prompt_dict
from openai import OpenAI
import yaml
import pprint
import requests
import numpy as np
import concurrent.futures
import time
import random
from prm import multiprocess_prompts

def get_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value][0]

def lang_detect(l):
    if len(l) in language_dict.keys():
        return l
    else:
        return get_keys_by_value(language_dict, l)

def dataset_language_detect(dataset, lang_type):
    if not lang_type:
        if dataset in ["OLAIR/mt-math-extended", "OLAIR/mt-aime-extended", "OLAIR/M-IMO-extended"]:
            return lang_dict[55]
        elif dataset in ["OLAIR/mt-math-500"]:
            return lang_dict[14]
        elif dataset in ["OLAIR/MMO"]:
            return lang_dict["MMO"]
        else:
            raise TypeError(f"Sorry! {dataset} does not suppoorted yet.")
    elif type(lang_type) == list:
        return lang_type
    else:
        return lang_dict[lang_type]

# ---------------------------------------------------------------------------
# Evaluate model for language using iterative PRM over each question
# ---------------------------------------------------------------------------
def evaluate_model_for_language( df, language, model_path, output_path, dataset, S=1, C=4, num_workers=4):
    col_name = lang_detect(language) if "MMO" not in dataset else "question"
    if col_name not in df.columns:
        print(f"Error: Language column '{col_name}' not found in dataset for language '{language}'. Skipping...")
        return

    # Create a list of prompts from the dataframe.
    prompts = list(df[col_name].values)
    print(f"Running iterative PRM for {len(prompts)} prompts (S={S}, C={C}) on language: {language}")

    # Process prompts in parallel.
    iterative_results = multiprocess_prompts(prompts, S, C, num_workers=num_workers)
    responses = [' '.join(res.get("steps", [])) if "steps" in res and res.get("steps") else "" for res in iterative_results]

    # Prepare output directory and filename.
    model_name = model_path.replace('/', '_')
    lang_clean = language.replace(" ", "_")
    out_dir = os.path.join(output_path, dataset_name_dict[dataset], model_name)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{lang_clean}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(len(df)):
            record = {
                "original_question": df[col_name].iloc[i],
                "question": df[col_name].iloc[i],
                "response": str(responses[i]),
                "answer": str(df["answer"].iloc[i]) if "answer" in df.columns else ""
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results for language '{language}' saved to {output_file}")

def main(models, datasets, lang_type, sample, output_dir, S=1, C=4, num_workers=4, max_model_len=4096):
    client = OpenAI()
    openai_models = [m.id for m in client.models.list().data]
    print(f"Loading model: {models[0]}")
    

    for output_path in output_dir:
        for data in datasets:
            if "MMO" in data:
                ds = load_dataset(data)
                df = ds['train'].to_pandas()  # assuming language splits exist inside 'train'
            else:
                df = load_dataset(data, split="train").to_pandas()
            if sample:
                df = df.sample(100, random_state=1210)
            for language in dataset_language_detect(data, lang_type):
                if "MMO" in data:
                    df = ds[language].to_pandas()
                out_file = os.path.join(output_path, dataset_name_dict[data], models[0].replace("/", "_"), f"{language}.jsonl")
                if os.path.exists(out_file):
                    continue
                print(f"Running model: {models[0]} for language: {language}")
                evaluate_model_for_language(df, language,  models[0], output_path, data, S=S, C=C, num_workers=num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    huggingface_hub.login(args["hf_token"])
    os.environ["OPENAI_API_KEY"] = args["openai_token"]
    
    # You can add S and C to your YAML config as needed.
    main(args["models"], args["datasets"], args["language_type"], args["samples"], args["output_path"],
         S=args.get("S", 1), C=args.get("C", 4), num_workers=args.get("num_workers", 4))
