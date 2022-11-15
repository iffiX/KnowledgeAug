import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from torch.distributions import Normal
from transformers import AutoModel, AutoTokenizer
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)


class RewardPredictor(nn.Module):
    def __init__(
        self,
        base_type,
        pad_by_longest: Optional[bool] = False,
        max_length: Optional[int] = None,
    ):
        super(RewardPredictor, self).__init__()
        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            truncation_side="left",
        )
        self.fc = nn.Linear(self.base.config.hidden_size, 1)
        self.pad_by_longest = pad_by_longest
        self.max_length = max_length

    @property
    def device(self):
        return self.fc.weight.device

    def forward(
        self,
        state: List[str],
        action: List[str],
        inference: bool = False,
        inference_batch_size: int = 128,
    ):
        assert len(state) == len(action), "Length of states and actions doesn't match"
        if not inference:
            batch_encoding = self.tokenizer(
                state,
                action,
                padding="max_length" if not self.pad_by_longest else "longest",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            # print(self.tokenizer.decode(batch_encoding["input_ids"][0]))
            return self.fc(self.base(**batch_encoding).last_hidden_state[:, 0, :])
        else:
            result = []
            with t.no_grad():
                # sort state and action by combined length
                all_data = [
                    (len(s) + len(a), idx, s, a)
                    for idx, (s, a) in enumerate(zip(state, action))
                ]
                all_data = sorted(all_data, key=lambda x: x[0])
                key = [d[1] for d in all_data]
                indices = [key.index(i) for i in range(len(state))]
                state = [d[2] for d in all_data]
                action = [d[3] for d in all_data]

                for start in range(0, len(state), inference_batch_size):
                    batch_encoding = self.tokenizer(
                        state[start : start + inference_batch_size],
                        action[start : start + inference_batch_size],
                        padding="max_length" if not self.pad_by_longest else "longest",
                        max_length=self.max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)
                    result.append(
                        self.fc(self.base(**batch_encoding).last_hidden_state[:, 0, :])
                    )
                return t.cat(result, dim=0)[indices]
