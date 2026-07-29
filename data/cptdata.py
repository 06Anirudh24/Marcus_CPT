from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch

def _get_bin(task_name: str, split: str):
    assert task_name in ['quality', 'rehersal', 'instruct']
    bin_data_dir = "/home/aa2886/Marcus_CPT/data/dataset/bins"
    implemented_quality_split = {
        'raw': f'{bin_data_dir}/quality_all-raw.bin', # Quality-llama-1.5M
        'raw_qwen': f'{bin_data_dir}/qwen_quality_raw.bin', # Quality-qwen-1.5M - by Marcus
        'raw10': f'{bin_data_dir}/quality_all-raw-10M.bin', # Quality-llama-10M - by Marcus
        'qwen_reconstructed_raw10': f'{bin_data_dir}/reconstructed_qwen-raw-10M.bin', # Quality-qwen-10M - reconstructed from Marcus data
        'qwen_reconstructed_rephrase': f'{bin_data_dir}/reconstructed_qwen-rephrase.bin', # Quality-qwen-10M - reconstructed from Marcus data
        'raw_rephrase': f'{bin_data_dir}/quality_all-raw-rephrase-llama3.bin', # Quality-llama-rephrase(10M by Marcus)
        'qwen-quality-rephrase': f'{bin_data_dir}/qwen_quality_rephrase.bin', # Quality-qwen-rephrase-6M (Anirudh)
        'fairytale_llama_rephrase': f'{bin_data_dir}/fairytale_all-rephrasellama-3.1-8b.bin', # Fairytale rephrase by Marcus (Llama) ~150K
        'fairytale_qwen_rephrase': f'{bin_data_dir}/reconstructed_qwen-fairytale-rephrase.bin', # Fairytale reconstructed rephrase by Anirudh
        'fairytale_llama_raw': f'{bin_data_dir}/fairytale_llama_raw.bin', # FT Llama by Anirudh (~42k)
        'fairytale_qwen_raw': f'{bin_data_dir}/fairytale_qwen_raw.bin', # FT Qwen by Anirudh (~42k)

        # FULL RAW DATA
        'quality_full_raw6_qwen': f'{bin_data_dir}/quality_full_raw6_qwen.bin',
        'quality_full_raw6_llama': f'{bin_data_dir}/quality_full_raw6_llama.bin',
        'quality_full_raw9_qwen': f'{bin_data_dir}/quality_full_raw9_qwen.bin',
        'quality_full_raw9_llama': f'{bin_data_dir}/quality_full_raw9_llama.bin',
        'quality_full_raw9_olmo': f'{bin_data_dir}/quality_full_raw9_olmo.bin',

    }
    implemented_rehersal_split = {
        'rpj-train': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
        'rpj-test': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_test.bin',
        'olmo-rpj-train': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_train_olmo.bin',
        'olmo-rpj-test': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_test_olmo.bin',
	    'qwen-rpj-train': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_train_qwen.bin',
        'qwen-rpj-test': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_test_qwen.bin'
    }
    implemented_instruct_split = {
        'ultrachat-train': f'{bin_data_dir}/ultrachat_train.bin',
        'ultrachat-test': f'{bin_data_dir}/ultrachat_test.bin'
    }
    if task_name == 'quality':
        assert split in implemented_quality_split
        return implemented_quality_split[split]
    elif task_name == 'rehersal':
        assert split in implemented_rehersal_split
        return implemented_rehersal_split[split]
    elif task_name == 'instruct':
        assert split in implemented_instruct_split
        return implemented_instruct_split[split]
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


class _MemmapDataset(Dataset):
    def __init__(self, block_size: int, bin_file: str, subsample_ratio: float):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode='r')
        self.ids = self.ids[:int(len(self.ids)*subsample_ratio)]

    def __len__(self):
        return int(len(self.ids)/self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i*self.block_size
        end_ind = (i+1)*self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(input_ids=torch.from_numpy(x_id).long(),
                    labels=torch.from_numpy(x_id).long())

class CPTDataset(_MemmapDataset):
    def __init__ (self, block_size: int, rehersal_rate: float,
                 subsample_ratio: float):
        assert rehersal_rate <= 1.0
        self.rehersal_rate = rehersal_rate
        # self.rehersal_data = _MemmapDataset(block_size, _get_bin('rehersal', 'qwen-rpj-train'), 1.0)
        self.rehersal_data = _MemmapDataset(block_size, _get_bin('rehersal', 'olmo-rpj-train'), 1.0)
	# self.rehersal_data = _MemmapDataset(block_size, _get_bin('rehersal', 'rpj-train'), 1.0)
        super().__init__(block_size,
                            # _get_bin('quality', 'rephrase'),
                            _get_bin('quality', 'quality_full_raw9_olmo'),
                            subsample_ratio)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehersal_rate:
            idx = np.random.randint(len(self.rehersal_data))
            return self.rehersal_data[idx]
        else:
            return super().__getitem__(i)

def get_task_data_module(task_name: str,
                         block_size: int,
                         rehersal_rate: float,
                         subsample_ratio: float,
                         **kwargs):
    if task_name == 'quality':
        train = CPTDataset(block_size, rehersal_rate, subsample_ratio)
        #val = _MemmapDataset(block_size, _get_bin('rehersal', 'qwen-rpj-test'), 1.0)
        val = _MemmapDataset(block_size, _get_bin('rehersal', 'olmo-rpj-test'), 1.0)
	# val = _MemmapDataset(block_size, _get_bin('rehersal', 'rpj-test'), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    if task_name == 'instruct':
        train = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-train'), 1.0)
        val = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-test'), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


if __name__ == '__main__':
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=True)
    #tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B", use_fast=True)
    tokenizer.model_max_length=2**20 # this is to hide the token_len>128K wraning

    block_size = 2048
    rehersal_rate = 0.1
    subsample_ratio = 1.0
    task_name = 'quality'
    data_module = get_task_data_module(task_name, block_size,
                                       rehersal_rate, subsample_ratio)
    for example in data_module['train_dataset']:
        print(tokenizer.decode(example['input_ids']))
        import pdb; pdb.set_trace()
