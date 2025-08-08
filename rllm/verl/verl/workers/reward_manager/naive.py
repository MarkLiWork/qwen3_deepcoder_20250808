# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ProcessPoolExecutor  # 修改：改为ProcessPoolExecutor
import threading
from typing import Dict, Any

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score

#from verl.workers.reward_manager import register
#from verl.workers.reward_manager.abstract import AbstractRewardManager


# 新增：将process_row移到模块级别
def process_row_global(args):
    """全局函数，可以被pickle序列化"""
    i, data_item_dict = args
    
    prompt_ids = data_item_dict['prompts']
    response_ids = data_item_dict['responses']
    attention_mask = data_item_dict['attention_mask']
    ground_truth = data_item_dict['ground_truth']
    data_source = data_item_dict['data_source']
    extra_info = data_item_dict['extra_info']
    tokenizer = data_item_dict['tokenizer']
    compute_score = data_item_dict['compute_score']
    num_examine = data_item_dict['num_examine']
    
    reward_extra_info = defaultdict(list)
    
    prompt_length = prompt_ids.shape[-1]
    valid_prompt_length = attention_mask[:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    
    valid_response_length = attention_mask[prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    
    # decode
    prompt_str = tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    
    score = compute_score(
        data_source=data_source,
        solution_str=response_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
    )
    
    if isinstance(score, dict):
        reward = score["score"]
        # Store the information including original reward
        for key, value in score.items():
            reward_extra_info[key].append(value)
    else:
        reward = score
    
    # 简化打印逻辑：只打印前几个样本
    if i < num_examine:
        print("[prompt]", prompt_str)
        print("[response]", response_str)
        print("[ground_truth]", ground_truth)
        if isinstance(score, dict):
            for key, value in score.items():
                print(f"[{key}]", value)
        else:
            print(f"[score]", score)
    
    return i, score, valid_response_length, reward_extra_info

#@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Thread-safe dict for tracking printed data sources
        print_lock = threading.Lock()
        
        # 修改：准备参数列表，打包所有需要的数据
        args_list = []
        for i in range(len(data)):
            data_item = data[i]
            data_item_dict = {
                'prompts': data_item.batch['prompts'],
                'responses': data_item.batch['responses'],
                'attention_mask': data_item.batch['attention_mask'],
                'ground_truth': data_item.non_tensor_batch["reward_model"]["ground_truth"],
                'data_source': data_item.non_tensor_batch[self.reward_fn_key],
                'extra_info': data_item.non_tensor_batch.get("extra_info", None),
                'tokenizer': self.tokenizer,
                'compute_score': self.compute_score,
                'num_examine': self.num_examine
            }
            args_list.append((i, data_item_dict))

        # Process items in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=24) as executor:  # 修改：改为ProcessPoolExecutor，减少worker数量
            results = list(executor.map(process_row_global, args_list))  # 修改：使用全局函数

        # Fill reward tensor with results
        reward_extra_info = {}
        for i, score, valid_response_length, sample_reward_extra_info in results:
            reward_tensor[i, valid_response_length - 1] = score
            # Update extra info at the list index
            for key, value in sample_reward_extra_info.items():
                if key not in reward_extra_info:
                    reward_extra_info[key] = [None] * len(data)
                reward_extra_info[key][i] = value
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor