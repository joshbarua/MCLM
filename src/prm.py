import argparse
import pprint
import requests
from openai import OpenAI
from litellm import batch_completion
import numpy as np
import concurrent.futures
import time
import random

# -----------------------------
# Helper: Format prompt for generation.
# -----------------------------
def format_prompt(prompt: str, steps: list) -> list:
    """
    Prepare the message list for the generation API. 
    If no steps have been generated, just include the initial prompt.
    Otherwise, add the steps (joined) to the assistant's message.
    """
    if len(steps) == 0:
        message = [
            {'role': 'system', 'content': "Please reason step by step, and put your final answer within \\boxed{}."},
            {'role': 'user', 'content': prompt}
        ]
    else:
        # The assistant's message contains all previous steps concatenated,
        # with an <extra_0> marker to indicate the next generation point.
        message = [
            {'role': 'system', 'content': "Please reason step by step, and put your final answer within \\boxed{}."},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': ' '.join(steps) + ' <extra_0>'}
        ]
    return message

# -----------------------------
# Helper: Score a candidate using the reward (pooling) API.
# -----------------------------
def score_candidate(message, model_name="Qwen/Qwen2.5-Math-PRM-72B"):
    """
    Send the candidate text (wrapped in the message format) to the pooling API.
    Returns the reward score (assumed to be in result['data'][0]['data'][0][1]).
    """
    payload = {
        "model": model_name,
        "messages": message
    }
    response = requests.post("http://64.139.209.3:30038/pooling", json=payload)
    result = response.json()
    score = result['data'][0]['data'][0][1]
    return score

# -----------------------------
# Main: Iterative chain-of-thought with reward scoring.
# -----------------------------
def iterative_prm(prompt: str, S: int, C: int):
    """
    Generate a chain-of-thought response iteratively.
    
    For each of S steps:
      1. Use batch_completion to generate C candidate responses (max_tokens=256).
      2. Score each candidate via the reward API.
      3. Select the candidate with the highest score.
      4. Append the selected candidate to the history.
      5. If the candidate includes '\\boxed' and it's not the final step, append a confirmation message.
    
    After S steps, if no candidate contained '\\boxed', append "(The final answer is)" and do one more inference.
    """
    steps_history = []
    found_boxed = False

    for step in range(S):
        # Prepare the current prompt using the history of steps.
        message = format_prompt(prompt, steps_history)
        
        # Generate C candidates using batch_completion.
        responses = batch_completion(
            model='openai/Qwen/Qwen2.5-Math-1.5B-Instruct',
            api_key='token-abc123',
            api_base="http://108.53.57.130:50195/v1",
            messages=[message] * C,  # generate C copies of the prompt
            max_tokens=256,
            temperature=0.7
        )
        # Extract candidate texts.
        candidates = [res.choices[0].message.content for res in responses]

        # Score each candidate.
        scores = []
        for cand in candidates:
            # For scoring, we pass the candidate appended to the original prompt.
            score_msg = format_prompt(prompt, steps_history + [cand])
            score_val = score_candidate(score_msg)
            scores.append(score_val)

        # Select the best candidate.
        best_idx = int(np.argmax(scores))
        best_candidate = candidates[best_idx].replace('<extra_0>','')

        # Append the best candidate to our steps history.
        steps_history.append(best_candidate)

        # Check if the candidate includes a boxed answer.
        if "\\boxed" in best_candidate:
            found_boxed = True
            # If we're not on the final step, add a confirmation message.
            if step < S - 1:
                confirm_msg = " let me confirm this answer and move on to the next step."
                steps_history[-1] += confirm_msg

    # If after S steps no boxed answer is found, do one final inference.
    if not found_boxed:
        final_message = format_prompt(prompt, steps_history + ["(The final answer is)"])
        final_response = batch_completion(
            model='openai/Qwen/Qwen2.5-Math-1.5B-Instruct',
            api_key='token-abc123',
            api_base="http://108.53.57.130:50195/v1",
            messages=[final_message],
            # min_tokens=248,
            max_tokens=128,
            temperature=0.7
        )
        final_candidate = final_response[0].choices[0].message.content
        steps_history.append(final_candidate)
    
    # Return the full conversation (initial prompt and all generated steps).
    full_response = {
        "initial_prompt": prompt,
        "steps": steps_history
    }
    return full_response


def process_prompt(prompt, S, C, max_retries=3):
    """
    Wrapper function to run iterative_prm with retries.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"Processing prompt (attempt {attempt+1}): {prompt[:30]}...")
            result = iterative_prm(prompt, S, C)
            return result
        except Exception as e:
            print(f"Error processing prompt: {e}")
            attempt += 1
            # Exponential backoff with jitter
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    # If we fail max_retries times, return an error message.
    return {"initial_prompt": prompt, "error": f"Failed after {max_retries} attempts."}


def multiprocess_prompts(prompts, S, C, num_workers=4):
    """
    Process a list of prompts in parallel using ProcessPoolExecutor.
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks.
        futures = {executor.submit(process_prompt, prompt, S, C): prompt for prompt in prompts}
        for future in concurrent.futures.as_completed(futures):
            prompt_text = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Prompt {prompt_text[:30]}... generated an exception: {exc}")
    return results
