import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

LLAMA_BIN = "/home/aa2886/Marcus_CPT/data/dataset/bins/fairytale_all-rephrasellama-3.1-8b.bin"
OUT_BIN   = "/home/aa2886/Marcus_CPT/data/dataset/bins/reconstructed_qwen-fairytale-rephrase.bin"

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
qwen_tok  = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", use_fast=True)

# Load llama token stream
n_tokens = os.path.getsize(LLAMA_BIN) // 4  # int32 = 4 bytes
llama_ids = np.memmap(LLAMA_BIN, dtype=np.int32, mode="r", shape=(n_tokens,))

llama_eos = llama_tok.eos_token_id
llama_bos = llama_tok.bos_token_id
qwen_eos  = qwen_tok.eos_token_id

# Convert chunk-by-chunk using llama EOS as boundary
qwen_ids_out = []
start = 0

for i in tqdm(range(n_tokens), desc="Converting"):
    if int(llama_ids[i]) == llama_eos:
        chunk_ids = llama_ids[start:i].tolist()
        start = i + 1

        # Strip BOS if it exists (common if encode() added special tokens)
        if llama_bos is not None and len(chunk_ids) > 0 and chunk_ids[0] == llama_bos:
            chunk_ids = chunk_ids[1:]

        text = llama_tok.decode(chunk_ids, skip_special_tokens=True)
        if not text:
            continue

        # Re-tokenize with Qwen
        ids = qwen_tok.encode(text, add_special_tokens=False)
        if qwen_eos is not None:
            ids.append(qwen_eos)
        qwen_ids_out.extend(ids)

# Optionally: handle any tail tokens after the last EOS (if your original bin ended mid-chunk)
if start < n_tokens:
    tail_ids = llama_ids[start:].tolist()
    if llama_bos is not None and len(tail_ids) > 0 and tail_ids[0] == llama_bos:
        tail_ids = tail_ids[1:]
    tail_text = llama_tok.decode(tail_ids, skip_special_tokens=True)
    if tail_text:
        ids = qwen_tok.encode(tail_text, add_special_tokens=False)
        if qwen_eos is not None:
            ids.append(qwen_eos)
        qwen_ids_out.extend(ids)

# Write Qwen bin
os.makedirs(os.path.dirname(OUT_BIN), exist_ok=True)
arr = np.memmap(OUT_BIN, dtype=np.int32, mode="w+", shape=(len(qwen_ids_out),))
arr[:] = np.array(qwen_ids_out, dtype=np.int32)
arr.flush()

print(f"Wrote {OUT_BIN} with {len(qwen_ids_out)} Qwen tokens")
