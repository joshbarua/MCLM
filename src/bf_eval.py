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
from prm import format_prompt
from bf import multiprocess_prompts  # multiprocess_prompts now uses budget forcing
from transformers import AutoTokenizer

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
            raise TypeError(f"Sorry! {dataset} is not supported yet.")
    elif type(lang_type) == list:
        return lang_type
    else:
        return lang_dict[lang_type]

# ---------------------------------------------------------------------------
# Evaluate model for language using budget forcing over each question
# ---------------------------------------------------------------------------
def evaluate_model_for_language(df, language, model_path, tokenizer, output_path, dataset,
                                max_budget, num_workers=4,
                                gen_model=None, gen_api_key=None, gen_api_base=None,
                                verbose=False):
    col_name = lang_detect(language) if "MMO" not in dataset else "question"
    if col_name not in df.columns:
        print(f"Error: Language column '{col_name}' not found in dataset for language '{language}'. Skipping...")
        return

    # Create a list of prompts from the dataframe.
    prompts = list(df[col_name].values)
    print(f"Running budget forcing for {len(prompts)} prompts on language: {language}")

    # Process prompts in parallel using our multiprocess_prompts wrapper for budget forcing.
    bf_results = multiprocess_prompts(
        prompts, max_budget,
        gen_model=gen_model,
        tokenizer=tokenizer,
        gen_api_key=gen_api_key,
        gen_api_base=gen_api_base,
        verbose=verbose,
        num_workers=num_workers
    )
    # Assume each result is a list of generated steps; join them into one response string.
    responses = [ ' '.join(res) if isinstance(res, list) else str(res) for res in bf_results ]

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
                "response": responses[i],
                "answer": str(df["answer"].iloc[i]) if "answer" in df.columns else ""
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Results for language '{language}' saved to {output_file}")

def main(models, datasets, tokenizer_name, lang_type, sample, output_dir,
         max_budget, num_workers=4, max_model_len=4096,
         gen_model=None, gen_api_key=None, gen_api_base=None, verbose=False):
    client = OpenAI()
    openai_models = [m.id for m in client.models.list().data]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Loading model: {models[0]}")
    
    for output_path in output_dir:
        for data in datasets:
            if "MMO" in data:
                ds = load_dataset(data)
                df = ds.to_pandas()  # assuming language splits exist inside 'train'
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
                evaluate_model_for_language(
                    df, language, models[0], tokenizer, output_path, data,
                    max_budget,
                    num_workers=num_workers,
                    gen_model=gen_model,
                    gen_api_key=gen_api_key,
                    gen_api_base=gen_api_base,
                    verbose=verbose
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    huggingface_hub.login(config["hf_token"])
    os.environ["OPENAI_API_KEY"] = config["openai_token"]

    # Call main() with parameters loaded from the YAML config.
    main(
        models=config["models"],
        datasets=config["datasets"],
        tokenizer_name=config["tokenizer"],
        lang_type=config["language_type"],
        sample=config["samples"],
        output_dir=config["output_path"],
        max_budget=config["max_budget"],
        num_workers=config.get("num_workers", 4),
        gen_model=config["gen_model"],
        gen_api_key=config["gen_api_key"],
        gen_api_base=config["gen_api_base"],
        verbose=config.get("verbose", False)
    )
