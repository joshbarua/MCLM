import argparse
import pprint
import requests
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time
import random
from transformers import AutoTokenizer
import multiprocessing
from functools import partial
from litellm import text_completion

def slice_at_last_line_break(text, max_length):
    """
    Slices the text so that its length is <= max_length,
    cutting at the final line break ('\n') that occurs before max_length.
    If no line break is found, simply returns the text sliced at max_length.
    """
    if len(text) <= max_length:
        return text
    # Find the last newline before max_length
    last_break = text.rfind('\n', 0, max_length)
    if last_break == -1:
        # If no newline found, slice at max_length
        return text[:max_length]
    return text[:last_break]


# -----------------------------
# Candidate generation API call.
# -----------------------------
def single_completion(message, model, api_key, api_base, temperature, max_tokens, min_tokens=250):
    """
    Sends a single message to the text completion API.
    """
    response = text_completion(
        model=model,
        api_key=api_key,
        api_base=api_base,
        prompt=message,
        temperature=temperature,
        max_tokens=max_tokens, min_tokens=min_tokens
    )
    return response

def batch_completion(messages, model, api_key, api_base, temperature=0, min_tokens=250,max_tokens=256):
    """
    Given a list of messages, concurrently sends them to the API using multiprocessing.
    The number of worker processes is set to the number of messages.
    
    Args:
        messages (list): A list of strings, each representing a message prompt.
        model (str): The model identifier.
        api_key (str): API key for authentication.
        api_base (str): Base URL for the API.
        temperature (float, optional): Temperature setting for text completion. Default is 0.
        max_tokens (int, optional): Maximum tokens to generate. Default is 100.
    
    Returns:
        list: A list of responses from the API corresponding to each message.
    """
    num_workers = len(messages)
    with multiprocessing.Pool(processes=num_workers) as pool:
        worker_func = partial(
            single_completion,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            min_tokens=min_tokens,
            max_tokens=max_tokens
        )
        responses = pool.map(worker_func, messages)
    return responses

# -----------------------------
# Helper: Format prompt for generation or reward scoring.
# -----------------------------
def format_prompt(prompt: str, steps: list, tokenizer: AutoTokenizer, is_reward: bool) -> str:
    """
    Prepare the message list for the generation API.
    
    If no steps have been generated, only include the initial prompt.
    Otherwise, add the steps (joined) to the assistant's message.
    
    For generation (is_reward=False): The assistant's message is formatted
      without an extra marker at the end.
    For reward scoring (is_reward=True): An extra marker (<extra_0>) is appended
      to indicate the generation point.
    
    The final message list is then passed through the tokenizer's chat template
    to produce a string prompt.
    """
    base_message = [
        {'role': 'system', 'content': "Please reason step by step, and put your final answer within \\boxed{}."},
        {'role': 'user', 'content': prompt}
    ]
    if len(steps) > 0:
        if is_reward:
            # For reward scoring, add an extra marker.
            assistant_content = ' '.join(steps) + " <extra_0>"
            base_message.append({'role': 'assistant', 'content': assistant_content})
            message = base_message
        else:
            assistant_content = ' '.join(steps)
            base_message.append({'role': 'assistant', 'content': assistant_content})
            # continue_final_message signals we are continuing a conversation.
            message = tokenizer.apply_chat_template(base_message, tokenize=False, continue_final_message=True)
    else:
        # When no steps exist yet, include the generation prompt.
        message = tokenizer.apply_chat_template(base_message, tokenize=False, add_generation_prompt=True)
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
# Logging helper.
# -----------------------------
def log_info(message: str, verbose: bool):
    if verbose:
        print(message)

# -----------------------------
# Main: Iterative chain-of-thought with PRM.
# -----------------------------
def iterative_prm(prompt: str, S: int, C: int,
                  gen_model: str, tokenizer: AutoTokenizer, gen_api_key: str, gen_api_base: str,
                  reward_model: str = "Qwen/Qwen2.5-Math-PRM-72B",
                  verbose: bool = False) -> dict:
    """
    Generate a chain-of-thought response iteratively.
    
    For each of S steps:
      1. Use batch_completion to generate C candidate responses (max_tokens=256) for candidate generation.
      2. Score each candidate via the reward API.
      3. Select the candidate with the highest score.
      4. Append the selected candidate to the history.
      5. If the candidate includes '\\boxed' and it's not the final step, stop early.
    
    After S steps, if no candidate contained '\\boxed', perform one final inference.
    
    Args:
        prompt (str): The initial prompt.
        S (int): Maximum number of iterative steps.
        C (int): Number of candidates to generate per step.
        gen_model (str): Model identifier for candidate generation.
        gen_api_key (str): API key for candidate generation.
        gen_api_base (str): Base URL for candidate generation API.
        reward_model (str, optional): Model identifier for the reward (pooling) API.
        verbose (bool, optional): If True, print debugging information.
        
    Returns:
        dict: A dictionary containing the original prompt and the list of steps (candidates) generated.
    """
    steps_history = []
    found_boxed = False

    for step in range(S):
        log_info(f"\nStep {step + 1} of {S}:", verbose)
        # Prepare the generation prompt (without reward marker).
        gen_message = format_prompt(prompt, steps_history, tokenizer, is_reward=False)
        log_info("Generation prompt:", verbose)
        log_info(gen_message, verbose)
        
        # Generate C candidates using batch_completion.
        responses = batch_completion(
            messages=[gen_message] * C,  # generate C copies of the prompt
            model=gen_model,
            api_key=gen_api_key,
            api_base=gen_api_base,
            temperature=0.7,
            max_tokens=256
        )

        # Extract candidate texts.
        # (Assuming the response object has the structure: response.choices[0].message.content)
        candidates = [res.choices[0].text for res in responses]
        log_info("\nGenerated Candidates:", verbose)
        for idx, cand in enumerate(candidates):
            log_info(f"Candidate {idx}: {cand}", verbose)

        # Score each candidate.
        scores = []
        for idx, cand in enumerate(candidates):
            # For scoring, format the prompt with the candidate appended,
            # and include the reward marker.
            score_msg = format_prompt(prompt, steps_history + [cand], tokenizer, is_reward=True)
            score_val = score_candidate(score_msg, model_name=reward_model)
            scores.append(score_val)
            log_info(f"Score for Candidate {idx}: {score_val}", verbose)
        
        # Select the best candidate.
        best_idx = int(np.argmax(scores))
        best_candidate = candidates[best_idx].replace('<extra_0>', '')
        log_info(f"Selected Candidate {best_idx}: {best_candidate}\n", verbose)
        steps_history.append(best_candidate)

        # If a boxed answer is found, stop early.
        if "\\boxed" in best_candidate:
            found_boxed = True
            log_info("Boxed answer found. Stopping early.", verbose)
            break

    # If no boxed answer was found after S steps, perform one final inference.
    if not found_boxed:
        log_info("No boxed answer found after maximum steps. Performing final inference.", verbose)
        steps_history.append("The final answer is")
        final_message = format_prompt(prompt, steps_history, tokenizer, is_reward=False)
        final_response = batch_completion(
            messages=[final_message],
            model=gen_model,
            api_key=gen_api_key,
            api_base=gen_api_base,
            temperature=0.7,
            min_tokens=4,
            max_tokens=128
        )
        final_candidate = final_response[0].choices[0].text
        log_info(f"Final Candidate: {final_candidate}", verbose)
        steps_history.append(final_candidate)
    
    full_response = {
        "initial_prompt": prompt,
        "steps": steps_history
    }
    return full_response


def process_prompt(prompt, S, C, gen_model, tokenizer, gen_api_key, gen_api_base, reward_model, verbose=False, max_retries=3):
    """
    Wrapper function to run iterative_prm with retries.
    Receives all necessary inputs for generation and reward scoring.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"\nProcessing prompt (attempt {attempt+1}): {prompt[:30]}...")
            result = iterative_prm(
                prompt, S, C,
                gen_model=gen_model,
                tokenizer=tokenizer,
                gen_api_key=gen_api_key,
                gen_api_base=gen_api_base,
                reward_model=reward_model,
                verbose=verbose
            )
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


# def multiprocess_prompts(prompts, S, C, gen_model, tokenizer, gen_api_key, gen_api_base, reward_model, verbose=False, num_workers=4):
#     """
#     Processes a list of prompts concurrently using ProcessPoolExecutor.
#     Each prompt is processed with the process_prompt wrapper.
#     """
#     results = [None] * len(prompts)
#     with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
#         # Submit tasks along with their original index.
#         futures = {
#             executor.submit(process_prompt, prompt, S, C, gen_model, tokenizer, gen_api_key, gen_api_base, reward_model, verbose): i
#             for i, prompt in enumerate(prompts)
#         }
#         for future in concurrent.futures.as_completed(futures):
#             idx = futures[future]
#             try:
#                 result = future.result()
#                 results[idx] = result
#             except Exception as exc:
#                 print(f"Prompt at index {idx} generated an exception: {exc}")
#                 results[idx] = {"initial_prompt": prompts[idx], "error": str(exc)}
#     return results

def multiprocess_prompts(prompts, S, C, gen_model, tokenizer, gen_api_key, gen_api_base, reward_model, verbose=False, num_workers=4):
    """
    Processes a list of prompts concurrently using ProcessPoolExecutor.
    Each prompt is processed with the process_prompt wrapper.
    """
    results = [None] * len(prompts)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks along with their original index.
        futures = {
            executor.submit(process_prompt, prompt, S, C, gen_model, tokenizer, gen_api_key, gen_api_base, reward_model, verbose): i
            for i, prompt in enumerate(prompts)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing prompts"):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as exc:
                print(f"Prompt at index {idx} generated an exception: {exc}")
                results[idx] = {"initial_prompt": prompts[idx], "error": str(exc)}
    return results
