import torch
from typing import Sequence, Optional, Union, Iterable
from vllm import LLM, SamplingParams

from utils import python_utils
from transformers import AutoTokenizer
from utils.prompt_utils import generate_all_answer_strings


def llama_forward(
    prefix_or_prefixes: Optional[Union[str, Iterable[str]]],
    prompts: Sequence[str],
    prefix: Optional[str],
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[str] = None,
    max_length: int = 300,  # Generation length
    temperature: float = 0.1,
    n_samples: int = 8,
    n_gpus: int = 2,
) -> Optional[Sequence[str]]:
    assert model is not None or model_path is not None, "model or model_path must be provided"

    if model is None:
        model = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=n_gpus)
    
    tokenizer = model.get_tokenizer()

    if temperature == 0.0:
        n_samples = 1

    formatted_prompts = []
    if isinstance(prefix_or_prefixes, str):
        for prompt in prompts:
            dialog = [
                {"role": "system", "content": prefix_or_prefixes},
                {"role": "user", "content": prefix + "\n" + prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(dialog, tokenize=False)
            formatted_prompts.append(formatted_prompt)
    elif isinstance(prefix_or_prefixes, Sequence):
        for prefix, prompt in python_utils.zip_(prefix_or_prefixes, prompts):
            dialog = [
                {"role": "system", "content": prefix},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(dialog, tokenize=False)
            formatted_prompts.append(formatted_prompt)
    elif prefix_or_prefixes is None:
        for prompt in prompts:
            dialog = [
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(dialog, tokenize=False)
            formatted_prompts.append(formatted_prompt)
    else:
        raise ValueError(
            f"prefix_or_prefixes must be None, a string, or a sequence of strings, not {type(prefix_or_prefixes)}")

    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        max_tokens=max_length,
        stop=[tokenizer.eos_token] + generate_all_answer_strings()
    )

    outputs = model.generate(prompts=formatted_prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            answer = ith_output.stop_reason
            if answer:
                attempts.append(ith_output.text + answer)
        result.append(attempts)

    return result