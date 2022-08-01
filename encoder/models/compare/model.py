import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from transformers import AutoModel
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)


class Model(nn.Module):
    """
    BERT style classifier, with VAT (Virtual adversarial training)

    Input format is [CLS] Question [SEP] Answer_A/B/C/D... [SEP]

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """

    def __init__(
        self, base_type, batch_size: int = 32,
    ):
        super(Model, self).__init__()

        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.linear = nn.Linear(self.base.config.hidden_size, 1)
        self.batch_size = batch_size

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        """

        Args:
            input_ids: LongTensor of shape [batch_size, compare_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, compare_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, compare_num, seq_length]
            labels: Binary LongTensor of shape [batch_size, compare_num]

        Returns:
            loss: Loss of training.
        """
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
        return loss

    def predict(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        compare_indices: List[List[Tuple[int, int]]] = None,
    ):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, compare_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, compare_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, compare_num, seq_length]
            compare_indices: A list of sub lists, where each sub list are made up of tuples,
                each tuple containing the left choice index and the right choice index. (When
                sigmoid logit < 0.5, left choice is chosen, otherwise right choice is chosen).

                When set to None, the compare indices are default to:
                [[(0, 1), (0, 2), ... (0, n-1), (1, 2), (1, 3),... (1, n-1), ..., (n-2, n-1)] ...]
        Returns:
            A list of chosen indexes
        """
        compare_num = input_ids.shape[1]
        sqrt = math.isqrt(1 + 8 * compare_num)
        assert (
            sqrt ** 2 == 1 + 8 * compare_num
        ), "compare number is not a valid combination of n*(n-1)/2"
        choice_num = (1 + sqrt) // 2
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        compare_indices = (
            compare_indices
            or [[(i, j) for i in range(choice_num) for j in range(i + 1, choice_num)]]
            * input_ids.shape[0]
        )
        result = []
        threshold = 0
        for logit, compare_index in zip(logits.to("cpu"), compare_indices):
            win_number = np.zeros([compare_num])
            for l, c in zip(logit, compare_index):
                if l < -threshold:
                    win_number[c[0]] += 1
                elif l > threshold:
                    win_number[c[1]] += 1
            result.append(np.argmax(win_number))

        return result

    def _forward(
        self, input_ids, attention_mask, token_type_ids, inputs_embeds=None,
    ):
        if inputs_embeds is None:
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            flat_input_ids = None

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        if flat_attention_mask.shape[0] >= self.batch_size:
            hidden_state = torch.zeros(
                [flat_token_type_ids.shape[0], self.base.config.hidden_size],
                device=flat_attention_mask.device,
            )
            for b in range(0, flat_attention_mask.shape[0], self.batch_size):
                model_input = {
                    "input_ids": flat_input_ids[b : b + self.batch_size]
                    if flat_input_ids is not None
                    else None,
                    "attention_mask": flat_attention_mask[b : b + self.batch_size],
                    "inputs_embeds": inputs_embeds[b : b + self.batch_size]
                    if inputs_embeds is not None
                    else None,
                }
                if token_type_ids is not None:
                    model_input["token_type_ids"] = flat_token_type_ids[
                        b : b + self.batch_size
                    ]
                hidden_state[b : b + self.batch_size] = self.base(
                    **model_input
                ).last_hidden_state[:, 0, :]

        else:
            model_input = {
                "input_ids": flat_input_ids,
                "attention_mask": flat_attention_mask,
                "inputs_embeds": inputs_embeds,
            }
            if token_type_ids is not None:
                model_input["token_type_ids"] = flat_token_type_ids
            hidden_state = self.base(**model_input).last_hidden_state[:, 0, :]

        return self.linear(hidden_state).view(input_ids.shape[0], input_ids.shape[1])
