import re
import copy
import random
import torch as t
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizerBase
from .openbook_qa import OpenBookQADataset


class OpenBookQAAugDataset(OpenBookQADataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
        similarity_embeddings: Tuple[List[str], t.Tensor, Dict[str, t.Tensor]],
        max_seq_length: int = 300,
        prepend_similar_training_example_num: int = 0,
        output_mode: str = "single",
    ):
        self.augment_contexts = augment_contexts
        self.similarity_embeddings = similarity_embeddings
        self.prepend_similar_training_example_num = prepend_similar_training_example_num
        self.rand = random.Random(42)
        self.rand_train = True
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

        data = copy.deepcopy(data)

        if split != "train":
            if "all of these" in data["choices"]:
                idx = data["choices"].index("all of these")
                other_idx = [x for x in range(len(data["choices"])) if x != idx]
                data["choices"][idx] = (
                    ", ".join(data["choices"][i] for i in other_idx[:-1])
                    + " and "
                    + data["choices"][other_idx[-1]]
                )
            elif "b and d" in data["choices"]:
                data["choices"][data["choices"].index("b and d")] = (
                    data["choices"][1] + " and " + data["choices"][3]
                )

        reference_examples = []
        if self.prepend_similar_training_example_num > 0:
            similarity = t.sum(
                self.similarity_embeddings[2][data["id"]]
                * self.similarity_embeddings[1],
                dim=1,
            )
            if split == "train":
                most_similar_train_indices = t.topk(
                    similarity, k=self.prepend_similar_training_example_num + 1,
                ).indices[1:]
            else:
                most_similar_train_indices = t.topk(
                    similarity, k=self.prepend_similar_training_example_num,
                ).indices
            for index in most_similar_train_indices:
                if similarity[index] > 0.62:
                    reference_examples.append(self.original_train_data[index])

        if self.output_mode == "single":
            encoded_sentence = self.tokenizer(
                self.normalize_t5_input(
                    ", ".join(self.get_augment_context(split, data["id"]))
                    + " \\n "
                    + ", ".join(
                        e["text_question"] + " " + e["text_answer"]
                        for e in reference_examples
                    )
                    + " \\n "
                    + data["text_question"]
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
            # Add choices has negative impact on accuracy
            # encoded_sentence = self.tokenizer(
            #     # [
            #     #     " [SEP] ".join(
            #     #         [
            #     #             ", ".join(
            #     #                 self.get_augment_context("train", r["id"], no_rand=True)
            #     #             )
            #     #             + r["text_question"]
            #     #             + "\\n "
            #     #             + r["text_choices"]
            #     #             + "\\n "
            #     #             + r["text_answer"]
            #     #             for r in reference_examples
            #     #         ]
            #     #     )
            #     # ]
            #     # * len(data["choices"]),
            #     # [
            #     #     ", ".join(self.get_augment_context(split, data["id"]))
            #     #     + data["text_question"]
            #     #     + "\\n "
            #     #     + data["text_choices"]
            #     #     + "\\n "
            #     #     + ch
            #     #     for ch in data["choices"]
            #     # ],
            #     [" ".join(self.get_augment_context(split, data["id"]))]
            #     * len(data["choices"]),
            #     [
            #         "; ".join(
            #             [
            #                 "question: "
            #                 + r["text_question"]
            #                 + " choices: "
            #                 + r["text_choices"]
            #                 + " answer: "
            #                 + r["text_answer"]
            #                 for r in reference_examples
            #             ]
            #             + [
            #                 self.normalize_question(data["text_question"])
            #                 + " "
            #                 # + ", ".join(data["choices"])
            #                 # + " [SEP] "
            #                 + ch
            #             ]
            #         )
            #         for ch in data["choices"]
            #     ],
            #     truncation="only_first",
            #     padding="max_length",
            #     max_length=self.max_seq_length,
            #     return_tensors="pt",
            # )

            encoded_sentence = self.tokenizer(
                # [
                #     " [SEP] ".join(
                #         [
                #             ", ".join(
                #                 self.get_augment_context("train", r["id"], no_rand=True)
                #             )
                #             + r["text_question"]
                #             + "\\n "
                #             + r["text_choices"]
                #             + "\\n "
                #             + r["text_answer"]
                #             for r in reference_examples
                #         ]
                #     )
                # ]
                # * len(data["choices"]),
                # [
                #     ", ".join(self.get_augment_context(split, data["id"]))
                #     + data["text_question"]
                #     + "\\n "
                #     + data["text_choices"]
                #     + "\\n "
                #     + ch
                #     for ch in data["choices"]
                # ],
                [", ".join(self.get_augment_context(split, data["id"]))]
                * len(data["choices"]),
                [
                    self.normalize_question(data["text_question"]) + " " + ch
                    for ch in data["choices"]
                ],
                truncation="only_first",
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            data["sentence"] = encoded_sentence.input_ids.unsqueeze(0)
            data["mask"] = encoded_sentence.attention_mask.unsqueeze(0)
            data["type_ids"] = encoded_sentence.token_type_ids.unsqueeze(0)
        return data

    def normalize_question(self, question):
        return re.sub(r"[^\w]\?$", "?", question).capitalize()

    def get_augment_context(self, split, data_id, no_rand=False):
        if split != "train":
            augment_context = self.augment_contexts[0].get(data_id, [])
        else:
            if self.rand_train and not no_rand:
                ref_context = self.augment_contexts[1].get(data_id)
                augment_context = self.augment_contexts[0].get(data_id, None)
                if augment_context is not None:
                    augment_context = self.rand.choice([ref_context, augment_context])
                else:
                    augment_context = ref_context
            else:
                augment_context = self.augment_contexts[1].get(data_id)
        return augment_context
