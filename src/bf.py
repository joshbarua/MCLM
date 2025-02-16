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
from prm import format_prompt

def log_info(message: str, verbose: bool):
    if verbose:
        print(message)
        
def single_completion(message, model, api_key, api_base, max_tokens, is_final=False):
    """
    Sends a single message to the text completion API.
    """
    if is_final:
        stop = []
    else:
        stop = ['</think>','**Final Answer**']
    response = text_completion(
        model=model,
        api_key=api_key,
        api_base=api_base,
        prompt=message,
        temperature=0.0,
        max_tokens=max_tokens,
        stop = stop
    )
    return response

def budget_forcing(prompt, max_budget, gen_model, tokenizer,gen_api_key, gen_api_base, verbose=False):
    steps_history = []
    found_boxed = False
    token_count = 0
    max_tokens = max_budget*0.9
    step = 0
    while token_count < max_budget:
        log_info(f"\nStep {step} Generated Token: {token_count}:", verbose)
        gen_message = format_prompt(prompt, steps_history, tokenizer, is_reward=False)
        log_info("Generation prompt:", verbose)
        log_info(gen_message, verbose)

        response = single_completion(
            message=gen_message,  # generate C copies of the prompt
            model=gen_model,
            api_key=gen_api_key,
            api_base=gen_api_base,
            max_tokens = int(max_tokens))
        response = response.choices[0].text
        steps_history.append(response)
        token_count+=len(tokenizer(response)['input_ids'])
        steps_history.append("Wait..let me try reconfirming")
        step+=1
        log_info(f"\nGenerated Response: {response}", verbose)

        if " ".join(steps_history).count("the original prompt is written in")>2:
            break
        if " ".join(steps_history).count("boxed")>3:
            break
        if " ".join(steps_history).count("Afrikaans")>3:
            break
        if " ".join(steps_history).lower().count("wait")>10:
            break
    steps_history.append("</think>\nTo summarize the final answer is")    
    response = single_completion(
            message=gen_message,  # generate C copies of the prompt
            model=gen_model,
            api_key=gen_api_key,
            api_base=gen_api_base,
            max_tokens = int(max_tokens),
            is_final = True    
        )
    response = response.choices[0].text
    steps_history.append(response)
    return " ".join(steps_history)


def process_prompt(prompt, max_budget, gen_model, tokenizer, gen_api_key, gen_api_base, verbose=False, max_retries=3):
    """
    Wrapper function to run budget_forcing with retries.
    Receives all necessary inputs for generation.
    
    Args:
        prompt (str): The initial prompt.
        max_budget (int): The maximum token budget.
        gen_model (str): The model identifier for generation.
        tokenizer: The tokenizer instance.
        gen_api_key (str): API key for generation.
        gen_api_base (str): Base URL for the generation API.
        verbose (bool): Enable verbose logging if True.
        max_retries (int): Maximum number of retry attempts.
    
    Returns:
        The generated steps (list) or an error message if all attempts fail.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"\nProcessing prompt (attempt {attempt+1}): {prompt[:30]}...")
            result = budget_forcing(
                prompt, max_budget,
                gen_model=gen_model,
                tokenizer=tokenizer,
                gen_api_key=gen_api_key,
                gen_api_base=gen_api_base,
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




def multiprocess_prompts(prompts, max_budget, gen_model, tokenizer, gen_api_key, gen_api_base, verbose=False, num_workers=4):
    """
    Processes a list of prompts concurrently using ProcessPoolExecutor.
    Each prompt is processed with the updated process_prompt wrapper which now uses max_budget.
    
    Args:
        prompts (list): List of prompt strings.
        max_budget (int): Maximum token budget for each prompt.
        gen_model (str): Generation model identifier.
        tokenizer: Tokenizer instance.
        gen_api_key (str): API key for generation.
        gen_api_base (str): Base URL for the generation API.
        verbose (bool): Enable verbose logging.
        num_workers (int): Number of parallel processes.
        
    Returns:
        list: A list of results corresponding to each prompt.
    """
    results = [None] * len(prompts)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks along with their original index.
        futures = {
            executor.submit(process_prompt, prompt, max_budget, gen_model, tokenizer, gen_api_key, gen_api_base, verbose): i
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
