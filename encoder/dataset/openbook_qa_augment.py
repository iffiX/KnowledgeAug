import random
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizerBase
from .openbook_qa import OpenBookQADataset


class OpenBookQAAugDataset(OpenBookQADataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
        max_seq_length: int = 300,
        output_mode: str = "single",
    ):
        self.augment_contexts = augment_contexts
        self.rand = random.Random(42)
        super(OpenBookQAAugDataset, self).__init__(
            tokenizer,
            max_seq_length=max_seq_length,
            use_matcher=False,
            output_mode=output_mode,
        )
        for split, split_data in (
            ("train", self.train_data),
            ("validate", self.validate_data),
            ("test", self.test_data),
        ):
            found_count = sum(
                1 for data in split_data if data["id"] in augment_contexts[0]
            )
            print(
                f"{found_count}/{len(split_data)} samples of {split} split have contexts"
            )

    def generator(self, index: int, split: str):
        if split == "train":
            data = self.train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        else:
            data = self.test_data[index]

        if split != "train":
            augment_context = self.augment_contexts[0].get(data["id"], [])
        else:
            # ref_context = self.augment_contexts[1].get(data["id"])
            # augment_context = self.augment_contexts[0].get(data["id"], None)
            # if augment_context is not None:
            #     augment_context = self.rand.choice([ref_context, augment_context])
            # else:
            #     augment_context = ref_context

            augment_context = self.augment_contexts[1].get(data["id"])

        if self.output_mode == "single":
            encoded_sentence = self.tokenizer(
                self.normalize_t5_input(
                    data["text_question"]
                    + " \\n "
                    + ", ".join(augment_context)
                    + " \\n "
                    + data["text_choices"].replace("\n", " ")
                ),
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            data["sentence"] = encoded_sentence.input_ids
            data["mask"] = encoded_sentence.attention_mask
            answer = self.tokenizer.encode(
                self.normalize_t5_input(data["text_answer"]),
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            )
            # Use -100 to focus on training the answer part, rather than pad
            # tokens
            answer.masked_fill_(answer == self.tokenizer.pad_token_id, -100)
            data["answer"] = answer
        else:
            encoded_sentence = self.tokenizer(
                [", ".join(augment_context)] * len(data["choices"]),
                [data["text_question"] + " " + ch for ch in data["choices"]],
                padding="max_length",
                max_length=self.max_seq_length,
                truncation="only_first",
                return_tensors="pt",
            )
            data["sentence"] = encoded_sentence.input_ids.unsqueeze(0)
            data["mask"] = encoded_sentence.attention_mask.unsqueeze(0)
            data["type_ids"] = encoded_sentence.token_type_ids.unsqueeze(0)
        return data
