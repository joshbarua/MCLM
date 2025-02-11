import argparse
import os
import json
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import huggingface_hub
from litellm import batch_completion
from prompts import lang_dict, language_dict, dataset_name_dict, system_prompt_dict
from openai import OpenAI
import yaml

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

def system_prompt_select(model, system_prompt_dict=system_prompt_dict):
    return system_prompt_dict[model] if model in list(system_prompt_dict.keys()) else 'Return your response in \\boxed{} format.'

def message_generate(model, query, tokenizer):
    if ("euler" in model) or ("ckpt" in model) or ("amphora" in model):
        message = [
            {'role': 'system', 'content': system_prompt_select(model)},
            {'role': 'user', 'content': query if query else ""},
            {'role': 'assistant', 'content': '<think>'},
        ]
        return tokenizer.apply_chat_template(message, tokenize=False, continue_final_message=True)
    elif ("R1-Distill" in model) or ("Eurus" in model):
        message = [
            {'role': 'user', 'content': "\n\n".join([query, system_prompt_select(model)]) if query else ""},
        ]
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    else:
        message = [
            {'role': 'system', 'content': system_prompt_select(model)},
            {'role': 'user', 'content': query if query else ""},
        ]
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

def evaluate_model_for_language(llm, tokenizer, df, language, sampling_params, model_path, output_path, dataset):
    # Determine the column name for the current language.
    # For English, assume the original column "problem" is used.
    col_name = lang_detect(language)
        
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
            temperature=1,
            max_tokens=int(4096*0.8)
        )
        responses = [output.choices[0].message.content for output in outputs]
        print("Completion Succeed!")
    else:
        # Prepare prompts for each question
        qrys = [message_generate(model_path, p, tokenizer) for p in df[col_name].values]

        # Generate responses for all questions
        print(qrys[0])
        outputs = llm.generate(qrys, sampling_params)
        responses = [output.outputs[0].text for output in outputs]

    # Prepare output directory and filename.
    model_name = model_path.replace('/', '_')
    # Replace spaces in language with underscores for the filename.
    lang_clean = language.replace(" ", "_")
    out_dir = os.path.join(output_path, dataset_name_dict[dataset], model_name)
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

def main(models, datasets, lang_type, sample, output_path, max_model_len=4096):
    # Set sampling parameters (note: ensure max_tokens is an integer)
    client = OpenAI()
    openai_models = [m.id for m in client.models.list().data]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=int(max_model_len * 0.8), stop=['</solution>'])

    for model in models:
        revision = model.split("***")[-1] if "***" in model else "main"
        model_name = model.split("***")[0].replace("/", "_")
        print(f"Loading model: {model_name} - {revision}")
        if model not in openai_models:
            tokenizer = AutoTokenizer.from_pretrained(model)
            llm = LLM(model=model, max_model_len=max_model_len, tensor_parallel_size=torch.cuda.device_count(), revision=revision)
        else:
            llm, tokenizer = model, None

        for data in datasets:
            # Load dataset (assuming the split "train" exists)
            if "MMO" in data:
                ds = load_dataset(data)
            else:
                df = load_dataset(data, split="train").to_pandas()
            if sample:
                df = df.sample(100,random_state=1210)  # Sample 100 example if dataset is large

            # For each model, evaluate all requested languages.
            for language in dataset_language_detect(data, lang_type):
                if "MMO" in data:
                    df = ds[language].to_pandas()
                if os.path.exists(os.path.join(output_path, dataset_name_dict[data], model.replace("/", "_"), f"{language}.jsonl")):
                    continue
                print(f"Running model: {model_name} - {revision} for language: {language}")
                evaluate_model_for_language(llm, tokenizer, df, language, sampling_params, model.split("***")[0], output_path, data)

if __name__ == "__main__":
    with open("eval.yaml", "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)

    huggingface_hub.login(args["hf_token"])
    os.environ["OPENAI_API_KEY"] = args["openai_token"]
    
    main(args["models"], args["datasets"], args["language_type"], args["samples"], args["output_path"])
