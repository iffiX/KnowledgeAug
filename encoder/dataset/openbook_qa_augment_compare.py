import os
import random
import numpy as np
import torch as t
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizerBase, BatchEncoding
from .openbook_qa import OpenBookQADataset
from encoder.utils.file import open_file_with_create_directories


class OpenBookQAAugCmpDataset(OpenBookQADataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
        max_seq_length: int = 300,
    ):
        self.augment_contexts = augment_contexts
        self.rand = random.Random(42)
        super(OpenBookQAAugCmpDataset, self).__init__(
            tokenizer, max_seq_length=max_seq_length, use_matcher=False,
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
            ref_context = self.augment_contexts[1].get(data["id"])
            augment_context = self.augment_contexts[0].get(data["id"], None)
            if augment_context is not None:
                augment_context = self.rand.choice([ref_context, augment_context])
            else:
                augment_context = ref_context

            # augment_context = self.augment_contexts[1].get(data["id"])

        if split != "train":
            choice_num = len(data["choices"])
            compare = [
                data["choices"][i]
                + f" {self.tokenizer.sep_token} "
                + data["choices"][j]
                for i in range(choice_num)
                for j in range(i + 1, choice_num)
            ]

        else:
            cmp_label = []
            compare = []
            correct_choice = data["label"]
            wrong_choices = [
                i for i in range(len(data["choices"])) if i != data["label"]
            ]
            for wrong_choice in wrong_choices:
                if self.rand.random() < 0.5:
                    # left one is correct
                    compare.append(
                        data["choices"][correct_choice]
                        + f" /n "
                        + data["choices"][wrong_choice]
                    )
                    cmp_label.append(0)
                else:
                    compare.append(
                        data["choices"][wrong_choice]
                        + f" /n "
                        + data["choices"][correct_choice]
                    )
                    cmp_label.append(1)

            # for i in range(len(data["choices"])):
            #     for j in range(i + 1, len(data["choices"])):
            #         if i == correct_choice or j == correct_choice:
            #             if self.rand.random() < 0.5:
            #                 # left one is correct
            #                 compare.append(
            #                     data["choices"][correct_choice]
            #                     + f" /n "
            #                     + data["choices"][j if i == correct_choice else i]
            #                 )
            #                 cmp_label.append(0)
            #             else:
            #                 compare.append(
            #                     data["choices"][j if i == correct_choice else i]
            #                     + f" /n "
            #                     + data["choices"][correct_choice]
            #                 )
            #                 cmp_label.append(1)
            #         else:
            #             compare.append(
            #                 data["choices"][i] + f" /n " + data["choices"][j]
            #             )
            #             cmp_label.append(0.5)

            data["cmp_label"] = t.Tensor([cmp_label])

        encoded_sentence = self.tokenizer(
            [", ".join(augment_context)] * len(compare),
            [data["text_question"] + f" /n " + cmp for cmp in compare],
            padding="max_length",
            max_length=self.max_seq_length,
            truncation="only_first",
            return_tensors="pt",
        )
        data["sentence"] = encoded_sentence.input_ids.unsqueeze(0)
        data["mask"] = encoded_sentence.attention_mask.unsqueeze(0)
        data["type_ids"] = encoded_sentence.token_type_ids.unsqueeze(0)
        return data

    def validate_choice(self, batch: BatchEncoding, choice: List[int]):
        """
        For use with a classifier model
        """
        ref_labels = batch["label"].cpu().numpy()

        for i in range(len(choice)):
            answer = ["A", "B", "C", "D"][choice[i]]
            ref_answer = ["A", "B", "C", "D"][batch["label"][i]]
            tokens = batch["sentence"][i]

            if tokens.dim() > 1:
                sentences = [
                    self.tokenizer.decode(to, skip_special_tokens=True) for to in tokens
                ]

                if answer != ref_answer:
                    for j, sentence in enumerate(sentences):
                        print(f"sentence {j}: [{sentence}] \n")
            else:
                sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)
                if answer != ref_answer:
                    print(f"sentence: [{sentence}] \n")

            if answer != ref_answer:
                print(f"answer: [{answer}] \n" f"ref_answer: [{ref_answer}] \n")
        return {"accuracy": float(np.sum(np.array(choice) == ref_labels)) / len(choice)}

    def generate_test_result_choice(self, choice: List[int], directory: str):
        with open_file_with_create_directories(
            os.path.join(directory, "openbook_qa.csv"), "w"
        ) as file:
            if len(choice) != len(self.test_data):
                raise ValueError(
                    f"Choice size {len(choice)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D"]
            for label, preprocessed in zip(choice, self.test_data):
                file.write(f"{preprocessed['id']},{answer_keys[label]}\n")
