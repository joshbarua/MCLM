import pandas as pd
from math_verify import parse, verify
from collections import Counter
from langdetect import detect
import re

def remove_math_expressions(text: str) -> str:
    # Remove inline math expressions \( ... \)
    text = re.sub(r'\\\(.*?\\\)', '', text, flags=re.DOTALL)
    # Remove block math expressions \[ ... \]
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    # Remove boxed expressions \boxed{ ... }
    text = re.sub(r'\\boxed{.*?}', '', text, flags=re.DOTALL)
    # Remove special characters, newlines, colons, and asterisks
    text = re.sub(r'[\n*:\\]', '', text)
    return text.strip()
    
lang = "English"

# lang = "French"
# lang_id = 'fr'
def get_score(input_path):
# df = pd.read_json(f'outputs/nvidia_AceMath-1.5B-Instruct/{lang}.jsonl',lines=True)
    df = pd.read_json(input_path,lines=True)
    
    # df = pd.read_json(f'outputs/models_1.5b-m_ckpt-{ckpt}/{lang}.jsonl',lines=True)
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
            
        
    return correct/len(df)