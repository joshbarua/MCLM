import numpy as np
from langdetect import detect_langs
from prompts import language_dict, dataset_name_dict
import re
import os
import argparse
import pandas as pd

def arg_check(root_path, models, datasets, languages):
    output_datasets = datasets if datasets else [d for d in os.listdir(root_path) if (".DS" not in d) and ("ipynb" not in d)]
    output_models = models if models else []
    if not models:
        for da in output_datasets:
            for model in [m for m in os.listdir(os.path.join(root_path, da)) if (".DS" not in m) and ("ipynb" not in m)]:
                if model not in model_check:
                    output_models.append(model)
    output_languages = []
    if languages:
        for l in languages:
            output_languages.append(get_keys_by_value(language_dict, l) if l in language_dict.keys() else l)
    else:
        output_languages = list(language_dict.values())
    return output_datasets, output_models, output_languages

def get_keys_by_value(d, target_value):
    return [key for key, value in d.items() if value == target_value][0]

def remove_math_expressions(text: str) -> str:
    """
    Removes mathematical expressions from the given text.
    """
    # Remove inline math expressions \( ... \)
    text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
    # Remove block math expressions \[ ... \]
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    # Remove boxed expressions \boxed{ ... }
    text = re.sub(r'\\boxed{.*?}', '', text, flags=re.DOTALL)
    # Remove inline math expressions $...$
    text = re.sub(r'\$.*?\$', '', text, flags=re.DOTALL)
    # Remove special characters, newlines, colons, and asterisks
    text = re.sub(r'[\n*:\\]', '', text)
    return text.strip()

def safe_lang_detect(text: str):
    """
    Safely detect languages in the text using langdetect.
    
    Returns:
        A list of detected language objects if successful, or None if detection fails.
    """
    try:
        return detect_langs(text)
    except Exception:
        return None

def get_score_from_text(text: str, tgt_lang: str) -> float:
    """
    Process the text by removing math expressions, detect languages, 
    and return the detection probability for the target language.
    
    Args:
        text: The input text.
        tgt_lang: The target language code (e.g., 'en', 'de').
    
    Returns:
        A float representing the probability that the text is in the target language,
        or 0.0 if the target language is not found.
    """
    cleaned_text = remove_math_expressions(text)
    detected_langs = safe_lang_detect(cleaned_text)
    if not detected_langs:
        return 0.0

    # Iterate over detected languages and return the probability for tgt_lang
    for lang_obj in detected_langs:
        # If the language object has attributes (lang, prob), use them directly.
        if hasattr(lang_obj, 'lang') and hasattr(lang_obj, 'prob'):
            if lang_obj.lang == tgt_lang:
                return float(lang_obj.prob)
        else:
            # Otherwise, try parsing its string representation "lang:prob"
            try:
                lang, prob = str(lang_obj).split(':')
                if lang == tgt_lang:
                    return float(prob)
            except Exception:
                continue
    return 0.0
    
def get_lcs_score(df, tgt_lang: str, is_think: bool) -> float:
    """
    Calculate the average language detection score for the target language over all rows in the DataFrame.
    
    If is_think is True, only the part of the response after '<solution>' is considered.
    
    Args:
        df: A DataFrame with a 'response' column.
        tgt_lang: The target language code.
        is_think: A boolean flag indicating whether to use the text after '<solution>'.
    
    Returns:
        The average probability score for the target language.
    """
    tgt_lang = tgt_lang if tgt_lang in language_dict.keys() else get_keys_by_value(language_dict, tgt_lang)
    scores = []
    for _, row in df.iterrows():
        response = row.response
        if not response:
            continue
        if is_think:
            # If '<solution>' exists, only use the part after it; otherwise, use an empty string.
            if "<solution>" not in response:
                continue
            response = response.split('<solution>', 1)[1].split("</solution>")[0] if '<solution>' in response else ""
        
        score = get_score_from_text(response, tgt_lang)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def main(models, datasets, languages, output_path):
    root_path = "results"
    datasets, models, languages = arg_check(root_path, models, datasets, languages)
    print(f"Evaluating languages: {languages}")
    os.makedirs(output_path, exist_ok=True)
    for dataset in datasets:
        if not os.path.exists(os.path.join(root_path, dataset_name_dict[dataset])):
            continue
        res = {"model": []}
        for model in models:
            rp = f"{root_path}/{dataset_name_dict[dataset]}/{model.replace('/', '_')}"
            if not os.path.exists(rp):
                continue
            else:
                res["model"].append(model)
            for ln in languages:
                if ln not in res.keys():
                    res[ln] = []
                if os.path.exists(os.path.join(rp, f"{ln}.jsonl")):
                    df = pd.read_json(os.path.join(rp, f"{ln}.jsonl"), lines=True)
                    print(f"{dataset} - {model} - {ln} Evaluation")
                    check = any(keyword in model for keyword in ("ckpt", "amphora", "euler"))
                    score = get_lcs_score(df, get_keys_by_value(language_dict, ln), check)
                    print(score)
                    res[ln].append(score)
                else:
                    print(f"{dataset} - {model} - {ln} Failed")
                    res[ln].append(None)
        res = pd.DataFrame(res)
        res.to_csv(f"{output_path}/{dataset_name_dict[dataset]}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None, help="List of model paths.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset identifier (e.g., 'amphora/m-aime-2024').")
    parser.add_argument("--languages", nargs="*", default=None, help="List of languages (e.g., Korean, Chinese, etc.).")
    parser.add_argument("--output_path", type=str, default="lcs_results", help="Path to save the output JSONL files.")
    args = parser.parse_args()
    
    main(args.models, args.datasets, args.languages, args.output_path)