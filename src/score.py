import pandas as pd
from math_verify import parse, verify
from collections import Counter
from langdetect import detect
from prompts import dataset_name_dict, language_dict
import argparse
import re
import os
    
def get_score(input_path):
    df = pd.read_json(input_path,lines=True)
    
    correct=0
    output = []
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
            output.append([text,answer,row.answer, gold, any([is_correct_math_verify,is_correct_rule]) ] )
        except:
            continue
            
    return correct/len(df)*100, output

def main(datasets, languages, output_dir="score_result"):
    root_path = "results"
    if not datasets:
        datasets = [d for d in os.listdir(root_path) if (".DS" not in d) and ("ipynb" not in d)]
    os.makedirs(output_dir, exist_ok=True)
    for data in datasets:
        dataset = dataset_name_dict[data] if data in dataset_name_dict.keys() else data
        if ("M-IMO" in dataset) or ("MMO" in dataset):
            print(f"{dataset} scoring does not supported. Please use mo_score.py!")
        else:
            print(f"{dataset} scoring is started.")
        model_list = [m for m in os.listdir(os.path.join(root_path, dataset)) if (".DS" not in m) and ("ipynb" not in m)]
        res = {"model": model_list}
        for model in model_list:
            lang_list = [f"{la}.jsonl" if la not in language_dict.keys() else f"{language_dict[la]}.jsonl" for la in languages] if languages else [l for l in os.listdir(os.path.join(root_path, dataset, model)) if (".DS" not in l) and ("ipynb" not in l)]
            if lang_list not in list(res.keys()):
                res = res | {la: [] for la in lang_list}
            for lang in lang_list:
                if os.path.exists(os.path.join(root_path, dataset, model, lang)):
                    res[lang].append(get_score(os.path.join(root_path, dataset, model, lang))[0])
                else:
                    res[lang].append(None)
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_dir, f"{dataset}.csv"), index=False)
        print(f"{dataset} scoring was ended! The result file is saved in {os.path.join(output_dir, dataset+'.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset list to evaluate.")
    parser.add_argument("--languages", nargs="*", default=None, help="Language list to evaluate.")
    args = parser.parse_args()
    main(args.datasets, args.languages)