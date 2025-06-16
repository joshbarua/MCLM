from openai import OpenAI
from dotenv import load_dotenv
import os
from google import genai
from google.genai.types import HttpOptions
from google.genai import types
import litellm
from litellm import batch_completion
_VLLM_ENGINES: dict[str, "LLM"] = {} # global engine cache

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
litellm_api_key = os.getenv('LITELLM_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Set OpenAI API key
openai_client = OpenAI(api_key=openai_api_key)

openai_models = {"gpt-4o-mini", "gpt-4o", "o1", "o3", "o3-mini", "o4", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1.nano"}
gemini_models = {"gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.5-pro-preview-06-05", "gemini-2.5-flash-preview-05-20"}

def _get_vllm_engine(model: str, tp: int | None = None, max_ctx: int = 16384) -> "LLM":
    """
    Lazily create–and cache–one high-throughput vLLM engine per model.
    * gpu_memory_utilization=0.9   → maximise KV-cache and avoid page-outs
    * enforce_eager=True           → CUDA graphs for lower latency
    * dtype='auto'                 → use model-recommended precision ( bf16 / fp16 )
    """
    from vllm import LLM, SamplingParams

    if model not in _VLLM_ENGINES:
        _VLLM_ENGINES[model] = LLM(
            model=model,                         # HF repo / local path
            trust_remote_code=True,              # needed for many OSS chat models
            dtype="auto",
            tensor_parallel_size=(tp or 1),
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            max_model_len=max_ctx,
        )
    return _VLLM_ENGINES[model]

def generate_response(prompts, model, max_completion_tokens=4096, temperature=0.0, reasoning_model=False, use_vllm=False): 
    try:
        if use_vllm:
            llm = _get_vllm_engine(model, tp=4)
            sparams = SamplingParams(
                temperature=temperature,
                max_tokens=max_completion_tokens,        
            )
            outputs = llm.generate(prompts, sparams)     
            outputs = [output.outputs[0].text for output in outputs]
        elif model in openai_models:
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature)
            outputs = [response.choices[i].message.content for i in range(len(prompts))]
        elif model in gemini_models:
            gemini_client = genai.Client(http_options=HttpOptions(api_version="v1"))
            #gemini_client = genai.Client(api_key=gemini_api_key)
            outputs = []
            for prompt in prompts:
                response = gemini_client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_completion_tokens),
                )
                outputs.append(response.text)
        elif "neulab" in model:
            api_base = "https://cmu.litellm.ai"              
            if not model.startswith("litellm_proxy/"):
                model = f"litellm_proxy/{model}"
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            out = batch_completion(
                api_key=litellm_api_key,
                api_base=api_base,
                model=model,
                messages=messages,
                temperature=temperature
            )
            outputs = [out["choices"][i]["message"]["content"] for i in range(len(prompts))]
        else:
            raise ValueError(f"Invalid model: {model}")
        if reasoning_model:
            reasoning_content = [out["choices"][i]["message"]["reasoning_content"] for i in range(len(prompts))]
            content = [out["choices"][i]["message"]["content"] for i in range(len(prompts))]
            return reasoning_content, content
        return outputs
    except Exception as e:
        print(f"Error: {e}")
        return [None] * len(prompts)

def generate_response_with_retries(prompts, model, start_tag=None, end_tag=None, num_retries=3, max_completion_tokens=4096, temperature=0.0, reasoning_model=False, use_vllm=False):
    base_temperature = temperature
    for attempt in range(num_retries + 1):
        current_temperature = base_temperature + (attempt * 0.1)
        try:
            if use_vllm:
                llm = _get_vllm_engine(model, tp=4)
                sparams = SamplingParams(
                    temperature=current_temperature,
                    max_tokens=max_completion_tokens,
                )
                outputs = llm.generate(prompts, sparams)
                outputs = [output.outputs[0].text for output in outputs]
            elif model in openai_models:
                messages = [{"role": "user", "content": prompt} for prompt in prompts]
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=current_temperature)
                outputs = [response.choices[i].message.content for i in range(len(prompts))]
            elif model in gemini_models:
                gemini_client = genai.Client(http_options=HttpOptions(api_version="v1"))
                #gemini_client = genai.Client(api_key=gemini_api_key)
                outputs = []
                for prompt in prompts:
                    response = gemini_client.models.generate_content(
                        model=model,
                        contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=current_temperature,
                        max_output_tokens=max_completion_tokens),
                    )
                    outputs.append(response.text)
            elif "neulab" in model:
                api_base = "https://cmu.litellm.ai"              
                if not model.startswith("litellm_proxy/"):
                    model = f"litellm_proxy/{model}"
                messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
                out = batch_completion(
                    api_key=litellm_api_key,
                    api_base=api_base,
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature)
                outputs = [out[i]["choices"][0]["message"]["content"] for i in range(len(prompts))]
            else:
                raise ValueError(f"Invalid model: {model}")
            if start_tag and end_tag and not reasoning_model:
                outputs = [output.split(start_tag)[1].split(end_tag)[0].strip() for output in outputs]
            if reasoning_model:
                reasoning_content = [out["choices"][i]["message"]["reasoning_content"] for i in range(len(prompts))]
                content = [out["choices"][i]["message"]["content"] for i in range(len(prompts))]
                return reasoning_content, content
            return outputs
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt == num_retries:
                print("Max retries reached. Returning None.")
                return [None] * len(prompts)
            
extract_statements_template = """Your job is to extract correct and self-contained logical statements from the provided thought trace. A statement is self-contained if it includes all the contextual information needed to determine whether it is True or False without referring to any external information.
- Example of a self-contained statement: "The square root of 144 is 12"
- Example of a statement that is not self-contained: "Joey had 10 apples, so Joey and Alice had 15 apples"
Each extracted statement must be no more than two sentences long and correct.
Beyond the requirements above, prioritize selecting statements that require complex mathematical reasoning and knowledge.
First reason through the task, then output a csv file with one column named "statement" and one row for each statement.
Here is the input question and thought trace for your task:
Input Question:
{question}
Thought Trace:
{trace}""" 

verify_answer_template = """You are an expert verifier. You will be given a problem, the ground truth solution(s), and a solution generated by a language model. Your task is to determine if the model’s solution matches any of the ground truth solutions, regardless of reasoning or correctness of steps. If it matches, it is correct. Your final answer must be strictly either True or False, enclosed in three backticks (```). Here are the details for your task:
Problem:
{problem}
Ground Truth Solution(s):
{gt_solution}
Model Solution:
{model_solution}
"""

extract_answer_template = """You will be given a solution to a math problem. Your task is to extract the boxed answer from the given text. If their is no boxed answer, analyze the solution and extract the final answer. Please enclose the final answer within three backticks (```).
Here is the solution:
{answer}"""

verify_correct_statement_template = """You will be given a logical statement. Your task is to verify the correctness of the statement. If the statement is True, return <answer>True</answer>. If the statement is False or lacks sufficient contextual information to determine whether it's True or False, return <answer>False</answer>. First reason through the task, then provide your final answer.
Here is the logical statement for your task:
{statement}"""

create_false_statement_template = """You will be given a correct logical statement. Your job is to minimally perturb the correct statement to create a false statement. First reason through the task, then return the false statement in the following format:
<false_statement>...false statement...</false_statement>
Here is the correct logical statement for your task:
{statement}"""

verify_false_statement_template = """You will be given a logical statement. Your task is to verify the falseness of the statement. If the statement is False, return <answer>True</answer>. If the statement is True or lacks sufficient contextual information to determine whether it's True or False, return <answer>False</answer>. First reason through the task, then provide your final answer.
Here is the logical statement for your task:
{statement}"""

translate_template = """Your job is to return a translated version of the English text.
* Translate to {target_lang}.
* The translation must be fluent, easy to read by native speakers.
* Do not solve the prompt translate it.
* You must preserve all details including math notations (latex) and code.
* The math notations and code must not be translated, keep it as is.
* Return your translation in the following format.
<translation>
...translated text...
</translation>
The following is the source text for you task:
<English>
{source_text}
</English>"""

translate_template_backticks = """Your job is to return a translated version of the English text.
* Translate to {target_lang}.
* The translation must be fluent, easy to read by native speakers.
* Do not solve the prompt translate it.
* You must preserve all details including math notations (latex) and code.
* The math notations and code must not be translated, keep it as is.
* Return your translation enclosed in three backticks.
The following is the source text for you task:
<English>
{source_text}
</English>"""

translate_template_english = """Your job is to translate the following text to English.
* The translation must be fluent, easy to read by native speakers.
* Do not solve the prompt translate it.
* You must preserve all details including math notations (latex) and code.
* The math notations and code must not be translated, keep it as is.
* Return your translation enclosed in three backticks.
The following is the source text for you task:
<source>
{source_text}
</source>"""

create_repair_template = """You will be given a math problem and a reasoning trace in {language}. Your task it to insert 5 to 10 self-repair statements uniformly throughout the reasoning trace. Self-repair statements verify the prior reasoning steps to repair any misunderstandings, errors, or gaps in the reasoning process. A good self-repair statement provides a clear explanation of the error and how it can be fixed. The self-repair statements must be in English and 2-4 sentences long. Do not reference spot tags in your self-repair statements. Your final answer should be a jsonl file where each line is a dictionary with two keys: "statement" with type string and "spot" with type int. This task is critical for the success of my research project.
Here is the math problem:
{problem}
Here is the reasoning trace:
{trace}
"""

choose_repair_template = """You will be given a math problem and a reasoning trace in {language}. Your task it to find the 5 to 10 most critical thoughts that reflect upon prior reasoning steps to (a) identify pitfalls in the reasoning process, (b) repair any misunderstandings, errors, or gaps in the reasoning process, and (c) verify the reasoning trajectory. Each thought is separated by two newlines. Your final answer should be a jsonl file where each line is a dictionary with two keys: "explanation" with type string and "spot" with type int. The value for "explanation" should explain why the thought helps repair and guide the reasoning process and "spot" should be the index preceding the thought.
Here is the math problem:
{problem}
Here is the reasoning trace:
{trace}
"""

SYSTEM_MESSAGE = "Please reason step by step, and put your final answer within \\boxed{}."

LONG_SYSTEM_MESSAGE = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

distill_prefix_template = "Please reason step by step, and put your final answer within \\boxed{{}}. You must think and answer only in {language}."

distill_suffix_template = "Okay, let me try to figure this out."