import numpy as np
from langdetect import detect_langs
import re
import os
import argparse
import pandas as pd

name_dict = {
    "amphora/m-math500": "math500",
    "OLAIR/mt-math-extended": "math100",
    "OLAIR/mt-aime-extended": "aime2024",
    "OLAIR/M-IMO-extended": "IMO"
}

lmap = {
    "Afrikaans": "af",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Catalan": "ca",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Finnish": "fi",
    "French": "fr",
    "Gujarati": "gu",
    "Hebrew": "he",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Kannada": "kn",
    "Korean": "ko",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Punjabi": "pa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Albanian": "sq",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Vietnamese": "vi",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw"
}

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
    scores = []
    for _, row in df.iterrows():
        response = row.response
        if is_think:
            # If '<solution>' exists, only use the part after it; otherwise, use an empty string.
            response = response.split('<solution>', 1)[1] if '<solution>' in response else ""
        
        score = get_score_from_text(response, tgt_lang)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def main(models, languages, output_path, dataset):
    root_path = "results"
    res = {"model": []}
    os.makedirs(output_path, exist_ok=True)
    for model in models:
        rp = f"{root_path}/{name_dict[dataset]}/{model.replace('/', '_')}"
        res["model"].append(model)
        for fp in os.listdir(rp):
            if fp.endswith("jsonl"):
                if fp.replace(".jsonl", "") not in res.keys():
                    res[fp.replace(".jsonl", "")] = []
                lang = lmap[fp[:-6]]
                df = pd.read_json(os.path.join(rp, fp), lines=True)
                print(f"{model} - {fp}")
                check = "euler" in model
                score = get_lcs_score(df,lang,check)
                print(score)
                res[fp.replace(".jsonl", "")].append(score)
    res = pd.DataFrame(res)
    res.to_csv(f"{output_path}/{name_dict[dataset]}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="List of model paths.")
    parser.add_argument("--languages", nargs="+", required=True, help="List of languages (e.g., Korean, Chinese, etc.).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL files.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset identifier (e.g., 'amphora/m-aime-2024').")
    args = parser.parse_args()
    
    main(args.models, args.languages, args.output_path, args.dataset)