import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT) 
from typing import List
import numpy as np
from transformers import AutoTokenizer
import random
import glob
from tqdm import tqdm
from utils.io_utils import jload

def get_tokenizer(tokenizer_model_name: str)-> AutoTokenizer:
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name, use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning
    return tokenizer

# Three variants - tokenize_file, tokenize_raw, tokenize_all
def tokenize_file(file_path: str) -> list[int]:
    all_ids = []

    books = jload(file_path)
    for book in books:
        # print(book)
        all_ids.extend(books[book]["summaries_dict"]["0"])
        all_ids.extend(books[book]["summaries_dict"]["1"])
        all_ids.append(books[book]["final_summary"])
    return all_ids

def tokenize_raw(file_path: str) -> list[int]:
    all_ids = []
    books = jload(file_path)
    for book in books:
        all_ids.extend(books[book])
    return all_ids

def tokenize_all(file_path: str, raw: str) -> list[int]:
    all_ids = []
    raw_books = jload(raw)
    for book in raw_books:
        all_ids.extend(raw_books[book]) 
    books = jload(file_path)
    for book in books:
        all_ids.extend(books[book]["summaries_dict"]["0"])
        all_ids.extend(books[book]["summaries_dict"]["1"])
        all_ids.append(books[book]["final_summary"])
    return all_ids

def tokenize_list(text_list: List[str]) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    # tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-8B")
    # tokenizer = get_tokenizer("Qwen/Qwen3-4B-Base")
    tokenizer = get_tokenizer("allenai/Olmo-3-1025-7B")
    all_ids = []
    for text in tqdm(text_list):
        if text:
            # ids = tokenizer.encode(text) # add_special_tokens=True to add BOS token
            # ids.append(tokenizer.eos_token_id) # add the end of text token
            ids = tokenizer.encode(text, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)

            all_ids.extend(ids)
    return all_ids

def write_to_memmap_single(ids: List[int], filename: str):
    filename = f'/home/aa2886/Marcus_CPT/data/dataset/bins/{filename}'
    print(f'Writing to {filename} with length {len(ids)}')
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()

def _glob_all_json(dir_name: str) -> List[str]:
    return glob.glob(f'{dir_name}/*.json') + glob.glob(f'{dir_name}/.*.json')

# Entry point
def tokenize_quality_graph():
    # all_text = tokenize_file("data/dataset/raw/llama3-1024-hier.json")
    # all_text = tokenize_raw("data/dataset/raw/quality_all_articles_chunked_1024.json")
    # write_to_memmap_single(tokenize_list(all_text), 'llama3-1024-hier.bin')
    # all_text = tokenize_all("data/dataset/summ/llama3-300-hier-v2.json","data/dataset/raw/quality_all_articles_chunked_1024.json")
    # write_to_memmap_single(tokenize_list(all_text), 'raw_hier2.bin')

    # all_text = tokenize_raw("/home/aa2886/Marcus_CPT/data/dataset/raw/quality_FULL_6x.json")
    # write_to_memmap_single(tokenize_list(all_text), 'quality_full_raw6_qwen.bin')
    all_text = tokenize_raw("/home/aa2886/Marcus_CPT/data/dataset/raw/quality_FULL_9x.json")
    write_to_memmap_single(tokenize_list(all_text), 'quality_full_raw9_olmo.bin')


if __name__ == '__main__':
    # Writing to data/dataset/bins/llama3-1024-hier.bin with length 1018843 (1M)
    tokenize_quality_graph()