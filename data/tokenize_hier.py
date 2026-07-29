import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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
        all_ids.append(raw_books[book]) 
    books = jload(file_path)
    for book in books:
        for level in books[book]["summaries_dict"]:
            all_ids.extend(books[book]["summaries_dict"][level])
        all_ids.append(books[book]["final_summary"])
    rephrase_books = jload("data/dataset/rephrase/llama3-300-rephrase.json")
    for book in rephrase_books:
        all_ids.extend(rephrase_books[book]["easy"])
        all_ids.extend(rephrase_books[book]["medium"])
        all_ids.extend(rephrase_books[book]["hard"])
            
    return all_ids

def tokenize_list(text_list: List[str]) -> List[int]:
    """
    Tokenize the text and return the tokenized text
    """
    random.shuffle(text_list)
    tokenizer = get_tokenizer("meta-llama/Meta-Llama-3-8B")
    all_ids = []
    for text in tqdm(text_list):
        if text:
            ids = tokenizer.encode(text) # add_special_tokens=True to add BOS token
            ids.append(tokenizer.eos_token_id) # add the end of text token
            all_ids.extend(ids)
    return all_ids

def write_to_memmap_single(ids: List[int], filename: str):
    filename = f'data/dataset/bins/{filename}'
    print(f'Writing to {filename} with length {len(ids)}')
    dtype = np.int32
    ids_arr = np.array(ids, dtype=dtype)
    arr_len = len(ids_arr)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    arr[:] = ids_arr
    arr.flush()

def _glob_all_json(dir_name: str) -> List[str]:
    return glob.glob(f'{dir_name}/*.json') + glob.glob(f'{dir_name}/.*.json')

def tokenize_quality_graph():
    all_text = tokenize_all("data/dataset/summ/llama3-300-hier-v2.json", "data/dataset/raw/quality_all_articles.json")
    write_to_memmap_single(tokenize_list(all_text), 'llama3-300-raw-hier-rephrase.bin')

if __name__ == '__main__':
    # Writing to data/dataset/bins/llama3-1024-hier.bin with length 1018843 (1M)
    tokenize_quality_graph()