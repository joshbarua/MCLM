import pandas as pd
from prompts import judge_template, language_dict
from math_verify import parse, verify
from collections import Counter
from langdetect import detect
from openai import OpenAI
from tqdm.auto import tqdm
from datasets import load_dataset
import argparse
import json
import re
import os

def contains_any(string, imo_ignore):
    return any(value in string for value in values)

def get_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value][0]

def data_collect(models, data_path, root_path, judge="gpt-4o-mini"):
    data = []
    for mod in models:
        languages = [f for f in os.listdir(os.path.join(root_path, data_path, mod)) if ".jsonl" in f]
        for lang in languages:
            df = pd.read_json(os.path.join(root_path, data_path, mod, lang), lines=True)
            for i,row in df.iterrows():
                try:
                    package = {
                    "custom_id": f"{data_path}-{mod}-{lang.replace('.jsonl', '')}-{i}", 
                    "method": "POST", 
                    "url": "/v1/chat/completions", 
                    "body": {
                        "model": judge, 
                        "messages": [
                            {"role": "system", "content": "You are a good judge."},
                            {"role": "user", "content": judge_template.replace("<math_question>", row["question"]).replace("<correct_answer>", row["answer"]).replace("<model_solution>", str(row["response"]))}
                        ],
                        "max_tokens": 4096,
                        "temperature": 0}}
                    data.append(package)
                except:
                    print(f"{data_path} - {mod} - {lang} - {i} response is empty!")

    os.makedirs("batch_data", exist_ok=True)
    with open(os.path.join("batch_data", f"{data_path}.jsonl"), "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"""The batch data file was saved in here: {os.path.join('batch_data', f'{data_path}.jsonl')}""")

def send_batch():
    client = OpenAI()
    result_ids = {}
    files = [f for f in os.listdir("batch_data") if ".jsonl" in f]
    if not files:
        return "File does not exist!"
    for file in files:
        batch_input_file = client.files.create(
            file=open(os.path.join("batch_data", file), "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"{file.replace('.jsonl', '')} - gpt-4o-mini Judge"
            }
        )
        a = list(client.batches.list(limit=1))[0].id
    
        print(f"{os.path.join('batch_data', file.replace('.jsonl', ''))} file was uploaded! - batch_id: {a}")

        result_ids[file.replace(".jsonl", "")] = a
    
    return result_ids

def receive_batch(file_ids, save_path="batch_result"):
    os.makedirs(save_path, exist_ok=True)
    client = OpenAI()
    for key, val in file_ids.items():
        output_file_id = client.batches.retrieve(val).output_file_id
        file_response = client.files.content(output_file_id).content
    
        with open(os.path.join(save_path, f"{key}_judge.jsonl"), 'wb') as file:
            file.write(file_response)
        
        print(f"Batch file was save completely. Saved here: {os.path.join(save_path, f'{key}_judge.jsonl')}")

def mo_get_score(root_path, save_path="score_result"):
    for file in [f for f in os.listdir(root_path) if ".jsonl" in f]:
        if "IMO" in file:
            dataset = load_dataset("OLAIR/M-IMO-extended", split="train").to_pandas()
        df = pd.read_json(os.path.join(root_path, file), lines=True)
        df["judge"] = [df.loc[i, "response"]["body"]["choices"][0]["message"]["content"] for i in range(len(df))]
        df["model"] = ["-".join(row.custom_id.split("-")[1:-2]) for _,row in df.iterrows()]
        df["language"] = [row.custom_id.split("-")[-2] for _,row in df.iterrows()]
        models, languages = sorted(set(list(df["model"]))), sorted(set(list(df["language"])))
        result_dict = {key: [] for key in ["model"] + languages}

        for model in models:
            result_dict["model"].append(model)
            subset = df[df["model"] == model]
            subset.reset_index(inplace=True, drop=True)
            for lang in languages:
                true, false = 0, 0
                subsub = subset[subset["language"] == lang]
                subsub.reset_index(inplace=True, drop=True)
                for i,row in subsub.iterrows():
                    if "IMO" in file:
                        if not dataset.loc[i, get_keys_by_value(language_dict, lang)]:
                            continue
                    if row.judge == "[[TRUE]]":
                        true += 1
                    else:
                        false += 1
    
                acc = true / (true + false) * 100
                result_dict[lang].append(acc)
                
        pd.DataFrame(result_dict).to_csv(os.path.join(save_path, f"{file.split('_')[0]}.csv"), index=False)
        print(f"The score file for {file.split('_')[0]} was saved.")

def main(models, datasets, score_type):
    root_path = "results"
    if score_type == "data_collect":
        data_list = datasets if datasets else ["IMO", "MMO"]
        for data in data_list:
            model_list = models if models else [m for m in os.listdir(os.path.join(root_path, data)) if (".DS" not in m) and ("ipynb" not in m)]
            data_collect(model_list, data, root_path)
    elif score_type == "send_batch":
        result_ids = send_batch()
        with open("batch_ids.json", "w") as f:
            json.dump(result_ids, f, indent=4)
    elif score_type == "receive_batch":
        with open("batch_ids.json", "r") as f:
            batch_ids = json.load(f)
        receive_batch(batch_ids)
    elif score_type == "score":
        mo_get_score("batch_result")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=["IMO", "MMO"])
    parser.add_argument("--score_type", type=str, required=True, help="""["data_collect", "send_batch", "receive_batch", "score"]""")
    args = parser.parse_args()

    main(args.models, args.datasets, args.score_type)