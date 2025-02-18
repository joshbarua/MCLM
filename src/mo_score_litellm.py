from litellm import batch_completion
from src.prompts import judge_template, language_dict, lang_dict
from datasets import load_dataset
import pandas as pd
import argparse
import os

def get_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value][0]


def format_data(path, data, model, languages):
    res = {
        "question": [],
        "response": [],
        "answer": [],
        "model": [],
        "language": [],
        "input_prompt": [],
        "judge": []
    }
    result_dict = {key: [] for key in languages}
    for lang in languages:
        if (lang == "Chinese_(Simplified)") and (data == "MMO"):
            lang = "Chinese"
        if os.path.exists(os.path.join(path, data, model, f"{lang}.jsonl")):
            df = pd.read_json(os.path.join(path, data, model, f"{lang}.jsonl"), lines=True)
            for _,row in df.iterrows():
                res["model"].append(model)
                res["language"].append(lang)
                res["question"].append(row["question"])
                res["response"].append(row["response"])
                res["answer"].append(row["answer"])
                res["input_prompt"].append([
                    {"role": "system", "content": "You are a good judge."},
                    {"role": "user", "content": judge_template.replace("<math_question>", str(row["question"])).replace("<correct_answer>", str(row["answer"])).replace("<model_solution>", str(row["response"]))}
                ])
        else:
            continue
    
    return res


def lang_process(res, data, languages):
    reuslt_dict = {}
    for lang in languages:
        if (lang in list(res["language"])) or ("Chinese" in lang):
            if ("Chinese" in lang) and ("MMO" in data):
                original_lang = lang
                lang = "Chinese"
            subset = res[res["language"] == lang]
            subset.reset_index(inplace=True, drop=True)
            true, false = 0, 0
            for i,row in subset.iterrows():
                if "IMO" in data:
                    if not dataset.loc[i, get_keys_by_value(language_dict, lang)]:
                        continue
                if row.judge == "[[TRUE]]":
                    true += 1
                else:
                    false += 1
            
            try:
                acc = true / (true + false) * 100
            except ZeroDivisionError as e:
                acc = 0
            if ("Chinese" in lang) and ("MMO" in data):
                result_dict[original_lang].append(acc)
            else:
                result_dict[lang].append(acc)
        else:
            result_dict[lang].append(None)
    
    return reuslt_dict


def main(path_list, models, datasets, languages, judge):
    if not path_list:
        raise ValueError("You should provide valid `path_list`.")

    if not datasets:
        datasets = ["IMO", "MMO"]
        
    save_path, log_path = "mo_result", "mo_logs"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    for path in path_list:
        for data in datasets:
            models = models if models else [m for m in os.listdir(os.path.join(path, data)) if (".DS" not in m) and ("ipynb" not in m)]
            if data == "IMO":
                dataset = load_dataset("OLAIR/M-IMO-extended", split="train").to_pandas()
            for model in models:
                res = format_data(path, data, model)
                languages = lang_dict[data] if not languages else languages

                print(f"{model} - {data} is Evaluating!")
                outputs = batch_completion(
                    messages=res["input_prompt"],
                    temperature=0,
                    max_tokens=4096,
                    model=judge
                )
                for output in outputs:
                    try:
                        res["judge"].append(output.choices[0].message.content)
                    except:
                        res["judge"].append("")

                res = pd.DataFrame(res)
                try:
                    res.to_csv(f"{log_path}/{data}-{model}_log.csv", index=False)
                except:
                    res.to_csv(f"{log_path}/{data}-{model}_log.csv", index=False, escapechar="\\")
                print(f"{data} Evaluation was done!")

                result = pd.DataFrame(lang_process(res, data, languages))
                result.to_csv(f"{save_path}/{data}-{model}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_list", nargs="*", type=list, help="Path list the model results are saved.")
    parser.add_argument("--models", nargs="*", type=list, help="Model list to evaluate.")
    parser.add_argument("--datasets", nargs="*", type=list, default=["IMO", "MMO"], help="Dataset list to evaluate.")
    parser.add_argumnet("--languages", nargs="*", type=list, default=lang_dict)
    parser.add_argument("--judge", type=str, default="gpt-4o-mini", help="Judge model to evaluate the model resonse.")
    args = parser.parse_args()

    main(args.path_list, args.models, args.datasets, args.languages, args.judge)