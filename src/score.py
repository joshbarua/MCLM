import pandas as pd
from math_verify import parse, verify
from collections import Counter
from langdetect import detect
from prompts import dataset_name_dict, language_dict
import argparse
import re
import os

def is_sublist(small, big):
    return any(small == big[i:i+len(small)] for i in range(len(big) - len(small) + 1))
    
def get_score(input_path):
    df = pd.read_json(input_path,lines=True)
    
    correct = 0
    output = {"response": [], "response_answer": [], "answer": [], "parsed_answer": [], "correctness": []}
    for _,row in df.iterrows():
        try:
            text = row.response
            answer = parse(text)
            gold = parse(str(row.answer))
            is_correct_math_verify = verify(gold, answer)
            is_correct_rule = any([str(a) in str(row.answer) for a in answer]+[str(row.answer) in str(a) for a in answer])
            
            if is_correct_math_verify:
                correct +=1
            elif is_correct_rule:
                correct +=1
            output["response"].append(text)
            output["response_answer"].append(answer)
            output["answer"].append(row.answer)
            output["parsed_answer"].append(gold)
            output["correctness"].append(any([is_correct_math_verify,is_correct_rule]))
        except:
            output["response"].append(text)
            output["response_answer"].append(None)
            output["answer"].append(None)
            output["parsed_answer"].append(None)
            output["correctness"].append(None)
            continue
            
    return correct / len(df) * 100, pd.DataFrame(output)

def main(datasets, languages, output_dir="score_result", log_dir="score_logs"):
    root_path = "results"
    if not datasets:
        datasets = [d for d in os.listdir(root_path) if (".DS" not in d) and ("ipynb" not in d)]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    for data in datasets:
        os.makedirs(os.path.join(log_dir, data), exist_ok=True)
        dataset = dataset_name_dict[data] if data in dataset_name_dict.keys() else data
        if ("IMO" in dataset) or ("MMO" in dataset):
            print(f"{dataset} scoring does not supported. Please use mo_score.py!")
            continue
        else:
            print(f"{dataset} scoring is started.")
        model_list = [m for m in os.listdir(os.path.join(root_path, dataset)) if (".DS" not in m) and ("ipynb" not in m)]
        res = {"model": []}
        for model in model_list:
            os.makedirs(os.path.join(log_dir, data, model), exist_ok=True)
            lang_list = [f"{la}.jsonl" if la not in language_dict.keys() else f"{language_dict[la]}.jsonl" for la in languages] if languages else [l for l in os.listdir(os.path.join(root_path, dataset, model)) if (".DS" not in l) and ("ipynb" not in l)]
            if not is_sublist([la.replace(".jsonl", "") for la in lang_list], list(res.keys())):
                res = res | {la.replace(".jsonl", ""): [] for la in lang_list}
            for lang in lang_list:
                if os.path.exists(os.path.join(root_path, dataset, model, lang)):
                    print(f"{model} - {lang} Scoring...")
                    score, outputs = get_score(os.path.join(root_path, dataset, model, lang))
                    res[lang.replace(".jsonl", "")].append(score)
                    outputs.to_csv(os.path.join(log_dir, data, model, f"{lang}.csv"), index=False)
                else:
                    res[lang.replace(".jsonl", "")].append(None)
            res["model"].append(model)
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_dir, f"{dataset}.csv"), index=False)
        print(f"{dataset} scoring was ended! The result file is saved in {os.path.join(output_dir, dataset+'.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset list to evaluate.")
    parser.add_argument("--languages", nargs="*", default=None, help="Language list to evaluate.")
    args = parser.parse_args()
    main(args.datasets, args.languages)