import os
import re
import copy
import json
import random
import difflib
import logging
import numpy as np
import torch as t
from typing import List, Tuple, Dict, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.download import CommonsenseQA
from encoder.dataset.matcher.commonsense_qa import CommonsenseQAMatcher
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.file import PickleCache, open_file_with_create_directories
from .base import StaticIterableDataset
from .utils import normalize_t5_input


class CommonsenseQABaseDataset:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        match_closest_when_no_equal: bool = True,
    ):
        self.tokenizer = tokenizer
        self.match_closest_when_no_equal = match_closest_when_no_equal

        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.matcher = CommonsenseQAMatcher(tokenizer=self.matcher_tokenizer)
        self.commonsense_qa = CommonsenseQA().require()

        with PickleCache(
            os.path.join(preprocess_cache_dir, "commonsense_qa.data"),
            self.generate_data,
        ) as cache:
            self.train_data = cache.data["train"]
            self.validate_data = cache.data["validate"]
            self.test_data = cache.data["test"]

        self.disable_dict = {"train": [], "validate": []}
        self.set_corpus()

    @property
    def train_dataset(self):
        return StaticIterableDataset(len(self.train_data), self.generator, ("train",),)

    @property
    def validate_dataset(self):
        return StaticIterableDataset(
            len(self.validate_data), self.generator, ("validate",)
        )

    @property
    def test_dataset(self):
        return StaticIterableDataset(len(self.test_data), self.generator, ("test",))

    def generator(self, index: int, split: str):
        """
        Needs to be overridden
        """
        return {}

    def validate_logits(self, batch: BatchEncoding, logits: t.Tensor):
        """
        For use with a classifier model
        """
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1)
        ref_labels = batch["label"].cpu().numpy()

        for i in range(labels.shape[0]):
            answer = ["A", "B", "C", "D", "E"][labels[i]]
            ref_answer = ["A", "B", "C", "D", "E"][batch["label"][i]]

            tokens = batch["sentence"][i]
            if tokens.dim() > 1:
                sentences = [
                    self.tokenizer.decode(tok, skip_special_tokens=True)
                    for tok in tokens
                ]
                if answer != ref_answer:
                    for j, sentence in enumerate(sentences):
                        print(f"sentence {j}: [{sentence}] \n")
            else:
                sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)
                if answer != ref_answer:
                    print(f"sentence {i}: [{sentence}] \n")
            if answer != ref_answer:
                print(f"answer: [{answer}] \n" f"ref_answer: [{ref_answer}]")

        return {"accuracy": float(np.sum(labels == ref_labels)) / labels.shape[0]}

    def validate_tokens(self, batch: BatchEncoding, tokens: t.Tensor):
        """
        For use with a generator model
        """
        total = tokens.shape[0]
        correct = 0
        approximately_correct = 0
        missing = 0
        answers = {}
        for i in range(tokens.shape[0]):
            answer = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
            ref_answer_tensor = batch["answer"][i]
            ref_answer_tensor.masked_fill_(
                ref_answer_tensor == -100, self.tokenizer.pad_token_id
            )
            ref_answer = self.tokenizer.decode(
                ref_answer_tensor, skip_special_tokens=True
            )
            sentence = self.tokenizer.decode(
                batch["sentence"][i], skip_special_tokens=True
            )
            answers[batch["id"][i]] = False
            if answer == ref_answer:
                correct += 1
                answers[batch["id"][i]] = True
            elif answer not in batch["choices"][i]:
                if self.match_closest_when_no_equal:
                    # Gestalt Pattern Matching
                    # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                    possible_matches = difflib.get_close_matches(
                        answer, batch["choices"][i], n=1
                    )
                    if len(possible_matches) == 0:
                        missing += 1

                    elif possible_matches[0] == ref_answer:
                        approximately_correct += 1
                        correct += 1
                        answers[batch["id"][i]] = True
                else:
                    missing += 1

            if answer != ref_answer:
                print(
                    f"sentence: [{sentence}] \n"
                    f"answer: [{answer}] \n"
                    f"ref_answer: [{ref_answer}]"
                )

        print(f"Missing ratio {float(missing) / total}")
        if self.match_closest_when_no_equal:
            print(f"Approximately correct ratio {float(approximately_correct) / total}")
        return {"accuracy": float(correct) / total}

    def generate_test_result_logits(self, logits: t.Tensor, directory: str):
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1).tolist()
        with open_file_with_create_directories(
            os.path.join(directory, "commonsense_qa.jsonl"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(
                    json.dumps(
                        {"id": preprocessed["id"], "answerKey": answer_keys[label]}
                    )
                )

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "commonsense_qa.jsonl"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(preprocessed["choices"]):
                    if answer == choice:
                        file.write(
                            json.dumps(
                                {"id": preprocessed["id"], "answerKey": answer_keys[i]}
                            )
                        )
                        break
                else:
                    missing += 1
                    print(
                        f"Missing answer, choices: {preprocessed['choices']}, "
                        f"answer: {answer}, using default A as answer."
                    )
                    file.write(json.dumps({"id": preprocessed["id"], "answerKey": "A"}))
        print(f"Missing ratio {float(missing)/len(self.test_data)}")

    def set_corpus(self):
        corpus = []
        for data in self.train_data:
            corpus.append(
                self.matcher.tokenizer.encode(
                    data["text_question"] + " " + data["text_choices"],
                    add_special_tokens=False,
                )
            )
        print("Corpus loaded, begin setting")
        self.matcher.matcher.set_corpus(corpus)

    def generate_data(self):
        return {
            "train": self.parse_data(self.commonsense_qa.train_path),
            "validate": self.parse_data(self.commonsense_qa.validate_path),
            "test": self.parse_data(self.commonsense_qa.test_path),
        }

    def parse_data(self, path):
        data = []
        logging.info(f"Parsing {path}")
        with open(path, "r") as file:
            for line in file:
                entry = json.loads(line)
                text_choices = self.generate_choice_str(
                    [ch["text"] for ch in entry["question"]["choices"]]
                )

                choices = [
                    f"{ch['text'].lower().strip(',')}"
                    for ch in entry["question"]["choices"]
                ]

                preprocessed = {
                    "text_question": entry["question"]["stem"],
                    "text_choices": text_choices,
                    "question_concept": entry["question"]["question_concept"],
                    "choices": choices,
                    "id": entry["id"],
                }
                if "answerKey" in entry:
                    # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                    label = [
                        i
                        for i, ch in enumerate(entry["question"]["choices"])
                        if ch["label"] == entry["answerKey"]
                    ][0]

                    preprocessed["label"] = label
                    preprocessed["text_answer"] = choices[label]

                data.append(preprocessed)
        return data

    def generate_choice_str(self, choices: List[str]):
        result = ""
        for label, choice in zip(self.generate_labels(), choices):
            if len(choice) > 0:
                result += label + " " + choice + " "
        return result

    def generate_labels(self):
        labels = []
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            labels.append(f"({char})")
        return labels


class CommonsenseQAAugmentDataset(CommonsenseQABaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
        max_seq_length: int = 300,
        output_mode: str = "single",
    ):
        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")

        self.augment_contexts = augment_contexts
        self.max_seq_length = max_seq_length
        self.output_mode = output_mode
        self.rand = random.Random(42)
        self.rand_train = True

        super(CommonsenseQAAugmentDataset, self).__init__(tokenizer)
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

        if self.output_mode == "single":
            encoded_sentence = self.tokenizer(
                normalize_t5_input(
                    ", ".join(self.get_augment_context(split, data["id"]))
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
                normalize_t5_input(data["text_answer"]),
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
                ref_context = self.augment_contexts[1].get(data_id, None)
                augment_context = self.augment_contexts[0].get(data_id, None)
                if augment_context is not None and ref_context is not None:
                    augment_context = self.rand.choice([ref_context, augment_context])
                elif ref_context is not None:
                    augment_context = ref_context
                elif augment_context is None and ref_context is None:
                    augment_context = []
            else:
                augment_context = self.augment_contexts[1].get(data_id, [])
        return augment_context
