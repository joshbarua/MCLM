import pandas as pd
from math_verify import parse, verify
from collections import Counter
from langdetect import detect
from prompts import dataset_name_dict, language_dict
import argparse
import re
import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

def generate_response(prompt, system_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=4096,
            temperature=0.0)
        output = response.choices[0].message.content
        return output
    except Exception as e:
        print(f"Error: {e}")
        return None

def is_sublist(small, big):
    return any(small == big[i:i+len(small)] for i in range(len(big) - len(small) + 1))
    
def get_score(input_path):
    df = pd.read_json(input_path,lines=True)
    
    correct = 0
    output = {"response": [], "response_answer": [], "answer": [], "parsed_answer": [], "correctness": []}
    for _,row in tqdm(df.iterrows(), total=len(df)):
        if int(row["correct"]) == 1:
            correct += 1
        else:
            continue
        '''try:
            text = row.response
            answer = parse(text)
            gold = parse(str(row.answer))
            is_correct_math_verify = verify(gold, answer)
            is_correct_rule = any([str(a) in str(row.answer) for a in answer]+[str(row.answer) in str(a) for a in answer])

            answer = text.split("<|begin_of_solution|>")[1].strip()
            # extract the boxed answer
            answer = re.search(r'\\boxed\{(.*)\}', answer).group(1)
            if not answer:
                is_correct = False
            else:
                #prompt = f"Here is the ground truth solution for the given problem: {row.answer}\n\nPlease analyze the following attempt and verify if the boxed answer is equivalent to the ground truth: {answer}"
                prompt = f"Here is the ground truth solution(s) for the given problem: {row.answer}\n\nPlease verify if the following answer is equivalent to any of the ground truth solutions: {answer}"
                response = generate_response(prompt, "You are an expert mathematical verifier. Your final answer should either be \\boxed{True} or \\boxed{False}.")
                if "\\boxed{True}" in response:
                    is_correct = True
                    correct += 1
                else:
                    is_correct = False

            if ((is_correct_math_verify or is_correct_rule) and not (is_correct)) or (not (is_correct_math_verify or is_correct_rule) and (is_correct)):
                print(f"Answer: {answer}")
                print(f"Gold: {row.answer}")
                print(f"Math Verify: {is_correct_math_verify}, Rule: {is_correct_rule}, LLM: {is_correct}")
                print("-"*100)
            
            #if is_correct_math_verify:
            #    correct +=1
            #elif is_correct_rule:
            #    correct +=1
            output["response"].append(text)
            output["response_answer"].append(answer)
            output["answer"].append(row.answer)
            output["parsed_answer"].append(gold)
            output["correctness"].append(is_correct)
        except:
            output["response"].append(text)
            output["response_answer"].append(None)
            output["answer"].append(None)
            output["parsed_answer"].append(None)
            output["correctness"].append(None)
            continue'''
            
    #return correct / len(df) * 100, pd.DataFrame(output)
    return correct / len(df) * 100

def main(root_path, models, datasets, languages, output_dir="score_result", log_dir="score_logs"):
    datasets = datasets if datasets else [d for d in os.listdir(root_path) if (".DS" not in d) and ("ipynb" not in d)]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    for data in datasets:
        models = models if models else [m for m in os.listdir(root_path, data) if (".DS" not in m) and ("ipynb" not in m)]
        os.makedirs(os.path.join(log_dir, data), exist_ok=True)
        #dataset = dataset_name_dict[data] if data in dataset_name_dict.keys() else data
        dataset = data
        #if ("IMO" in dataset) or ("MMO" in dataset):
        #    print(f"{dataset} scoring does not supported. Please use mo_score.py or mo_score_lite_llm.py!")
        #    continue
        #else:
        print(f"{dataset} scoring is started.")
            
        res = {"model": []}
        for model in models:
            os.makedirs(os.path.join(log_dir, data, model), exist_ok=True)
            lang_list = [f"{la}.jsonl" if la not in language_dict.keys() else f"{language_dict[la]}.jsonl" for la in languages] if languages else [l for l in os.listdir(os.path.join(root_path, dataset, model)) if (".DS" not in l) and ("ipynb" not in l)]
            if not is_sublist([la.replace(".jsonl", "") for la in lang_list], list(res.keys())):
                res = res | {la.replace(".jsonl", ""): [] for la in lang_list}
            for lang in lang_list:
                print(os.path.join(root_path, dataset, model, lang))
                if os.path.exists(os.path.join(root_path, dataset, model, lang)):
                    print(f"{model} - {lang} Scoring...")
                    #score, outputs = get_score(os.path.join(root_path, dataset, model, lang))
                    score = get_score(os.path.join(root_path, dataset, model, lang))
                    res[lang.replace(".jsonl", "")].append(score)
                    #outputs.to_csv(os.path.join(log_dir, data, model, f"{lang.replace('.jsonl', '')}.csv"), index=False)
                else:
                    res[lang.replace(".jsonl", "")].append(None)
            res["model"].append(model)
        res = pd.DataFrame(res)
        res.to_csv(os.path.join(output_dir, f"{dataset}.csv"), index=False)
        print(f"{dataset} scoring was ended! The result file is saved in {os.path.join(output_dir, dataset+'.csv')}")

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--root_path", type=str, default="results", help="Root path the model responses results are saved.")
    #parser.add_argument("--models", nargs="*", default=None, help="Model list to evlauate.")
    #parser.add_argument("--datasets", nargs="*", default=["math100", "aime2024", "math500"], help="Dataset list to evaluate.")
    #parser.add_argument("--languages", nargs="*", default=None, help="Language list to evaluate.")
    #args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    with open("score.yaml", "r", encoding="utf-8") as file:
        args = yaml.safe_load(file)
    main(args["root_path"], args["models"], args["datasets"], args["languages"])