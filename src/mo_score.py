import pandas as pd
from prompts import judge_template
from math_verify import parse, verify
from collections import Counter
from langdetect import detect
from openai import OpenAI
from tqdm.auto import tqdm
import json
import re
import os

imo_ignore = ['Afrikaans-3',
 'Afrikaans-4',
 'Afrikaans-5',
 'Afrikaans-8',
 'Afrikaans-22',
 'Indonesian-0',
 'Indonesian-1',
 'Indonesian-2',
 'Italian-22',
 'Japanese-3']

def contains_any(string, imo_ignore):
    return any(value in string for value in values)


def mo_get_score(input_path, save_path):
    df = pd.read_json(input_path, lines=True)
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
                if (row.custom_id.split("_")[0] == "M-IMO"):
                    if contains_any(row.custom_id):
                        continue
                if row.judge == "[[TRUE]]":
                    true += 1
                else:
                    false += 1

            acc = true / (true + false) * 100
            result_dict[lang].append(acc)
            
    result = pd.DataFrame(result_dict)
    return result

def data_collect(model, root_path):
    models = [m for m in os.listdir(root_path) if "DS" not in m]
    data = []
    for mod in models:
        languages = [f for f in os.listdir(os.path.join(root_path, mod)) if ".jsonl" in f]
        for lang in languages:
            df = pd.read_json(f"{root_path}/{mod}/{lang}", lines=True)
            for i in range(len(df)):
                package = {
                "custom_id": f"{root_path.split('_')[-1]}-{mod}-{lang.replace('.jsonl', '')}-{i}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": model, 
                    "messages": [
                        {"role": "system", "content": "You are a good judge."},
                        {"role": "user", "content": judge_template.replace("<math_question>", df.loc[i, "question"]).replace("<correct_answer>", df.loc[i, "answer"]).replace("<model_solution>", df.loc[i, "response"])}
                    ],
                    "max_tokens": 4096,
                    "temperature": 0}}
                data.append(package)

    os.makedirs("batch_api", exist_ok=True)
    with open(f"batch_api/{root_path.split('_')[-1]}.jsonl", "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")

    return f"batch_api/{root_path.split('_')[-1]}.jsonl"

def send_batch(model, root_path):
    data_path = data_collect(model, root_path)
    client = OpenAI()
    batch_input_file = client.files.create(
        file=open(data_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"{root_path.split('_')[-1]} {model} Judge"
        }
    )

    print("Batch was uploaded!")

    return batch_input_file_id

def receive_batch(file_id, root_path, save_path="batch_result"):
    os.makedirs(save_path, exist_ok=True)
    client = OpenAI()
    
    file_response = client.files.content(file_id).content

    print("Batch file was loaded completely.")

    saving = f"{save_path}/{root_path.split('_')[-1]}_batch_result.jsonl"
    with open(saving, 'wb') as file:
        file.write(file_response)
    
    print(f"Batch file was save completely.\nSaved here: {saving}")