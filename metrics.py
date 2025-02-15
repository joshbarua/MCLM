import itertools
from sklearn.metrics import cohen_kappa_score
from typing import Any
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse
from math_verify import verify
from sympy import simplify


def compare_strings(text, answer):
    """
    Given a predicted answer (text) and the gold answer (answer),
    parses both using the provided extraction configurations and
    returns whether they verify as equal.
    """
    gold_parsed = parse(answer, [ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)])
    pred_parsed = parse(text, [ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)])
    return verify(gold_parsed, pred_parsed, 6, True)

def fleiss_kappa(ratings):
    """
    Compute Fleiss' kappa for a set of ratings.
    
    Parameters:
      ratings: A list of lists (or 2D numpy array) with shape (N, k) where:
               - N is the number of items,
               - k is the number of categories.
               Each row should contain the count of ratings in each category for that item.
    
    Returns:
      Fleiss' kappa score.
    """
    ratings = np.array(ratings)
    N, k = ratings.shape
    n = np.sum(ratings[0])  # number of raters per item (assumed constant)
    
    # Compute the extent of agreement for each item.
    P_i = np.sum(ratings * (ratings - 1), axis=1) / (n * (n - 1))
    P_bar = np.mean(P_i)
    
    # Compute the overall proportion of all assignments to each category.
    p_j = np.sum(ratings, axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)
    
    return (P_bar - P_e) / (1 - P_e)

def evaluate_consistency(data_path, print_image=False):
    """
    Evaluate the consistency of responses across multiple language JSONL files
    located in data_path. This function computes pairwise Cohen's kappa, displays
    a heatmap, and calculates the overall Fleiss' kappa for multi-rater agreement.
    """
    # --- Step 1: Load and Process the Data ---
    
    dataframes = {}
    for filename in os.listdir(data_path):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(data_path, filename)
            df = pd.read_json(filepath, lines=True)
            key = os.path.splitext(filename)[0]
            dataframes[key] = df

    n_rows = None
    for key, df in dataframes.items():
        if n_rows is None:
            n_rows = len(df)
        elif len(df) != n_rows:
            raise ValueError(f"DataFrame for {key} has {len(df)} rows; expected {n_rows} rows.")

    results = []
    for idx in tqdm(range(n_rows)):
        row_data = {'index': idx}
        for lang, df in dataframes.items():
            response = df.iloc[idx]['response']
            answer = df.iloc[idx]['answer']
            is_correct = compare_strings(str(response), str(answer))
            row_data[lang] = is_correct
        # Optional: mark as consistent if all languages agree
        row_data['consistent'] = (all(row_data[lang] for lang in dataframes.keys()) or
                                   not any(row_data[lang] for lang in dataframes.keys()))
        results.append(row_data)

    consistency_df = pd.DataFrame(results)

    # --- Step 2: Compute Pairwise Cohen's Kappa and Create a Heatmap ---

    language_columns = [col for col in consistency_df.columns if col not in ['index', 'consistent']]
    kappa_matrix = pd.DataFrame(index=language_columns, columns=language_columns, dtype=float)

    for lang1 in language_columns:
        for lang2 in language_columns:
            if lang1 == lang2:
                kappa_matrix.loc[lang1, lang2] = 1.0
            else:
                # Pass the labels parameter to ensure both classes (True, False) are considered.
                score = cohen_kappa_score(consistency_df[lang1], consistency_df[lang2], labels=[False, True])
                kappa_matrix.loc[lang1, lang2] = score
    if print_image:
        plt.figure(figsize=(10, 8))
        sns.heatmap(kappa_matrix.astype(float), annot=False, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Pairwise Cohen's Kappa Heatmap")
        plt.show()

    # --- Step 3: Compute Multi-Rater Agreement using Fleiss' Kappa ---

    ratings = []
    num_languages = len(language_columns)
    for _, row in consistency_df.iterrows():
        correct_count = sum(1 for lang in language_columns if row[lang])
        incorrect_count = num_languages - correct_count
        ratings.append([correct_count, incorrect_count])

    overall_fleiss_kappa = fleiss_kappa(ratings)
    print(f"Overall Fleiss' kappa (multi-rater agreement): {overall_fleiss_kappa:.3f}")

    return overall_fleiss_kappa

# Example usage:
# evaluate_consistency('7b/001/math100/Qwen_Qwen2.5-Math-7B-Instruct/')



def extract_equations(text):
    # Define individual regex patterns.
    patterns = [
        r'\\\(\s*(.*?)\s*\\\)',             # inline LaTeX: \( ... \)
        r'\\\\\(\s*(.*?)\s*\\\\\)',         # inline LaTeX with double backslashes: \\( ... \\)
        r'\\\[\s*(.*?)\s*\\\]',             # display math with single backslashes: \[ ... \]
        r'\\\\\[\s*(.*?)\s*\\\\\]',         # display math with double backslashes: \\[ ... \\]
        # Plain text equations containing an "=".
        # Negative lookbehind/ahead prevents matching when in the middle of a word.
        r'(?<![A-Za-z])([0-9A-Za-z\^\*\+\-/\\\(\)\[\]\{\}\s]+=[0-9A-Za-z\^\*\+\-/\\\(\)\[\]\{\}\s]+)(?![A-Za-z])'
    ]
    
    # Collect all matches with their positions.
    matches = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.DOTALL):
            matches.append((m.start(), m.end(), m.group(0)))
    
    # Sort matches by starting index.
    matches.sort(key=lambda x: x[0])
    
    # Merge matches that are likely part of the same equation,
    # but do NOT merge if the gap contains newlines or bullet markers.
    merged = []
    if matches:
        current_start, current_end, current_text = matches[0]
        for start, end, match_text in matches[1:]:
            gap = text[current_end:start]
            # Merge only if the gap is only whitespace or punctuation (no newline or bullet marker).
            if "\n" not in gap and re.fullmatch(r'[\s,;:\-]*', gap):
                current_end = end
                current_text = current_text.rstrip() + " " + match_text.lstrip()
            else:
                merged.append((current_start, current_end, current_text.strip()))
                current_start, current_end, current_text = start, end, match_text
        merged.append((current_start, current_end, current_text.strip()))
    
    # Post-process: if a merged segment still contains multiple sub-equations,
    # split by newline or bullet markers.
    final_equations = []
    for _, _, eq in merged:
        # Split by newline followed by a bullet marker (-, *, etc.)
        if "\n" in eq:
            parts = re.split(r'\n[-*]\s+', eq)
            for part in parts:
                part = part.strip()
                if len(part) > 5:
                    try:
                        final_equations.append(parse(part)[0])
                    except:
                        continue
        else:
            try:
                final_equations.append(parse(eq)[0])
            except:
                continue 
    return final_equations

def eq_equal(e1, e2):
    """
    Checks whether two sympy expressions are equivalent.
    Returns True if simplify(e1 - e2) equals 0, False otherwise.
    """
    try:
        diff = simplify(e1 - e2)
        return diff == 0
    except Exception:
        return False

def deduplicate(seq):
    """
    Returns a new list where equivalent sympy expressions have been removed,
    preserving the order of first occurrences.
    """
    unique = []
    for expr in seq:
        if not any(eq_equal(expr, unique_expr) for unique_expr in unique):
            unique.append(expr)
    return unique

def lcs_length(seq1, seq2):
    """
    Computes the length of the longest common subsequence (LCS) between two sequences of sympy expressions.
    Two expressions are considered matching if eq_equal returns True.
    Returns a tuple (lcs_length, dp) where dp is the DP table.
    """
    m = len(seq1)
    n = len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if eq_equal(seq1[i], seq2[j]):
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[m][n], dp

def pure_overlap(seq1, seq2):
    """
    Computes the number of overlapping equations (pure set intersection)
    between two sequences of sympy expressions after deduplication.
    Returns a tuple (overlap_count, normalized_overlap)
    where normalized_overlap is the ratio of common unique equations
    to the length of the larger deduplicated list.
    """
    dedup_seq1 = deduplicate(seq1)
    dedup_seq2 = deduplicate(seq2)
    count = 0
    for expr1 in dedup_seq1:
        if any(eq_equal(expr1, expr2) for expr2 in dedup_seq2):
            count += 1
    normalized = count / max(len(dedup_seq1), len(dedup_seq2)) if max(len(dedup_seq1), len(dedup_seq2)) > 0 else 0
    return count, normalized

def sequence_alignment_metrics(seq1, seq2):
    """
    Given two sequences of sympy expressions (i.e. equations), this function first deduplicates each list and then computes:
      1. overlap_count (order-sensitive LCS length)
      2. order_similarity (LCS normalized by max length)
      3. pure_overlap_count (set intersection size ignoring order)
      4. pure_overlap_similarity (normalized pure overlap)
    Returns a tuple with these four values.
    """
    # Deduplicate each list
    dedup_seq1 = deduplicate(seq1)
    dedup_seq2 = deduplicate(seq2)
    
    # Compute order-sensitive overlap (LCS)
    lcs_len, _ = lcs_length(dedup_seq1, dedup_seq2)
    order_similarity = lcs_len / max(len(dedup_seq1), len(dedup_seq2)) if max(len(dedup_seq1), len(dedup_seq2)) > 0 else 0

    # Compute pure overlap (set intersection)
    pure_overlap_count, pure_overlap_similarity = pure_overlap(seq1, seq2)
    
    return lcs_len, order_similarity, pure_overlap_count, pure_overlap_similarity


import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from metrics import extract_equations, sequence_alignment_metrics

def process_row(idx, dataframes, anchor_lang):
    """
    Process a single row index: extract equations from the anchor (English) response
    and from every other language response, then compute the four metrics.
    Returns a dict mapping each non-anchor language to its metrics for this row.
    """
    anchor_response = dataframes[anchor_lang].iloc[idx]['response']
    anchor_eqs = extract_equations(anchor_response)
    row_metrics = {}
    for lang, df in dataframes.items():
        if lang == anchor_lang:
            continue
        response = df.iloc[idx]['response']
        eqs = extract_equations(response)
        lcs_count, order_sim, pure_overlap_count, pure_overlap_sim = sequence_alignment_metrics(anchor_eqs, eqs)
        row_metrics[lang] = {
            "lcs": lcs_count,
            "order": order_sim,
            "pure_count": pure_overlap_count,
            "pure_sim": pure_overlap_sim
        }
    return row_metrics

def evaluate_equation_consistency(data_path, anchor_lang='english', print_image=False):
    """
    Evaluate the consistency of chain-of-thought equations across multiple language JSONL files.
    The English version (or a specified anchor language) is used as the anchor.
    For each row, equations are extracted and compared using sequence alignment metrics.
    This version uses multiprocessing to parallelize the row-level computations.
    """
    # --- Step 1: Load and Process the Data ---
    dataframes = {}
    for filename in os.listdir(data_path):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(data_path, filename)
            df = pd.read_json(filepath, lines=True)
            # Use the lowercase file name (without extension) as the language key
            key = os.path.splitext(filename)[0].lower()
            dataframes[key] = df

    # Ensure all dataframes have the same number of rows
    n_rows = None
    for key, df in dataframes.items():
        if n_rows is None:
            n_rows = len(df)
        elif len(df) != n_rows:
            raise ValueError(f"DataFrame for {key} has {len(df)} rows; expected {n_rows} rows.")

    if anchor_lang not in dataframes:
        raise ValueError(f"Anchor language '{anchor_lang}' not found in data files.")

    # --- Step 2: Parallel Processing of Each Row ---
    # Initialize accumulators for each non-anchor language and overall.
    language_metrics = { lang: {"lcs": [], "order": [], "pure_count": [], "pure_sim": []}
                         for lang in dataframes if lang != anchor_lang }
    overall_metrics = {"lcs": [], "order": [], "pure_count": [], "pure_sim": []}

    # Create a partial function that includes the fixed arguments.
    process_func = partial(process_row, dataframes=dataframes, anchor_lang=anchor_lang)

    # Use a multiprocessing pool to process rows in parallel.
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Using imap to preserve order; tqdm adds a progress bar.
        results = list(tqdm(pool.imap(process_func, range(n_rows)), total=n_rows, desc="Evaluating rows"))

    # --- Step 3: Accumulate Results ---
    for row_metric in results:
        for lang, metrics in row_metric.items():
            language_metrics[lang]["lcs"].append(metrics["lcs"])
            language_metrics[lang]["order"].append(metrics["order"])
            language_metrics[lang]["pure_count"].append(metrics["pure_count"])
            language_metrics[lang]["pure_sim"].append(metrics["pure_sim"])

            overall_metrics["lcs"].append(metrics["lcs"])
            overall_metrics["order"].append(metrics["order"])
            overall_metrics["pure_count"].append(metrics["pure_count"])
            overall_metrics["pure_sim"].append(metrics["pure_sim"])

    # --- Step 4: Compute Averages ---
    avg_language_metrics = {}
    for lang, metrics in language_metrics.items():
        avg_language_metrics[lang] = {
            "avg_lcs": np.mean(metrics["lcs"]),
            "avg_order": np.mean(metrics["order"]),
            "avg_pure_count": np.mean(metrics["pure_count"]),
            "avg_pure_sim": np.mean(metrics["pure_sim"]),
        }
    overall_avg = {
        "avg_lcs": np.mean(overall_metrics["lcs"]),
        "avg_order": np.mean(overall_metrics["order"]),
        "avg_pure_count": np.mean(overall_metrics["pure_count"]),
        "avg_pure_sim": np.mean(overall_metrics["pure_sim"]),
    }

    print("Per-language average metrics (compared to the anchor language):")
    for lang, avg in avg_language_metrics.items():
        print(f"{lang}: {avg}")
    print("Overall average metrics:", overall_avg)

    return avg_language_metrics, overall_avg

# # Example usage:
# if __name__ == '__main__':
#     data_path = "path/to/your/jsonl/files"  # Adjust to your actual path
#     evaluate_equation_consistency(data_path, anchor_lang='english')
