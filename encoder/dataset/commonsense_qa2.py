import os
import copy
import json
import random
import difflib
import logging
import numpy as np
import torch as t
from typing import List, Tuple, Dict, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.download import CommonsenseQA2
from encoder.dataset.matcher.commonsense_qa2 import CommonsenseQA2Matcher
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.file import PickleCache, open_file_with_create_directories
from .base import StaticIterableDataset
from .utils import normalize_t5_input


class CommonsenseQA2BaseDataset:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        match_closest_when_no_equal: bool = True,
    ):
        self.tokenizer = tokenizer
        self.match_closest_when_no_equal = match_closest_when_no_equal

        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.matcher = CommonsenseQA2Matcher(tokenizer=self.matcher_tokenizer)
        self.commonsense_qa2 = CommonsenseQA2().require()

        with PickleCache(
            os.path.join(preprocess_cache_dir, "commonsense_qa2.data"),
            self.generate_data,
        ) as cache:
            self.original_train_data = cache.data["train"]
            self.train_data = cache.data[
                "train"
            ]  # [d for d in cache.data["train"] if d["confidence"] >= 0.5]
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
            answer = ["yes", "no"][labels[i]]
            ref_answer = ["yes", "no"][batch["label"][i]]

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
        choices = [normalize_t5_input(x) for x in ["(A)", "(B)"]]
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
            elif answer not in choices:
                if self.match_closest_when_no_equal:
                    # Gestalt Pattern Matching
                    # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                    possible_matches = difflib.get_close_matches(answer, choices, n=1)
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
            os.path.join(directory, "commonsense_qa2.txt"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["yes\n", "no\n"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(answer_keys[label])

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        choices = [normalize_t5_input(x) for x in ["(A)", "(B)"]]
        with open_file_with_create_directories(
            os.path.join(directory, "commonsense_qa2.txt"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["yes\n", "no\n"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(choices):
                    if answer == choice:
                        file.write(answer_keys[i])
                        break
                else:
                    missing += 1
                    print(
                        f"Missing answer, choices: {choices}, "
                        f"answer: {answer}, using default yes as answer."
                    )
                    file.write("yes\n")
        print(f"Missing ratio {float(missing)/len(self.test_data)}")

    def set_corpus(self):
        corpus = []
        for data in self.train_data:
            corpus.append(
                self.matcher.tokenizer.encode(
                    data["text_question"], add_special_tokens=False,
                )
            )
        print("Corpus loaded, begin setting")
        self.matcher.matcher.set_corpus(corpus)

    def generate_data(self):
        return {
            "train": self.parse_data(self.commonsense_qa2.train_path),
            "validate": self.parse_data(self.commonsense_qa2.validate_path),
            "test": self.parse_data(self.commonsense_qa2.test_path),
        }

    def parse_data(self, path):
        data = []
        logging.info(f"Parsing {path}")
        with open(path, "r") as file:
            for line in file:
                entry = json.loads(line)
                text_choices = self.generate_choice_str(["yes", "no"])

                choices = ["yes", "no"]

                preprocessed = {
                    "text_question": entry["question"],
                    "text_choices": text_choices,
                    "choices": choices,
                    "id": entry["id"],
                    "confidence": entry["confidence"],
                }
                if "answer" in entry:
                    # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                    label = [
                        i for i, ch in enumerate(choices) if ch == entry["answer"]
                    ][0]
                    preprocessed["validations"] = entry["validations"]
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


class CommonsenseQA2AugmentDataset(CommonsenseQA2BaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        use_augment: bool,
        augment_contexts: Dict[str, List[str]],
        max_seq_length: int = 300,
        output_mode: str = "single",
    ):
        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")

        self.use_augment = use_augment
        self.augment_contexts = augment_contexts
        self.max_seq_length = max_seq_length
        self.output_mode = output_mode
        self.rand = random.Random(42)
        self.rand_train = True

        super(CommonsenseQA2AugmentDataset, self).__init__(tokenizer)
        for split, split_data in (
            ("train", self.train_data),
            ("validate", self.validate_data),
            ("test", self.test_data),
        ):
            found_count = sum(
                1 for data in split_data if data["id"] in augment_contexts
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
            t5_input = normalize_t5_input(
                (
                    ", ".join(self.augment_contexts.get(data["id"], []))
                    if self.use_augment
                    else ""
                )
                + " \\n "
                + data["text_question"]
                + " \\n "
                + data["text_choices"].replace("\n", " ")
            )
            encoded_sentence = self.tokenizer(
                t5_input,
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )

            answer = self.tokenizer.encode(
                normalize_t5_input(["(A)", "(B)"][data["label"]]),
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            )
            # Use -100 to focus on training the answer part, rather than pad
            # tokens
            answer.masked_fill_(answer == self.tokenizer.pad_token_id, -100)

            data["sentence"] = encoded_sentence.input_ids
            data["mask"] = encoded_sentence.attention_mask
            data["answer"] = answer
            data["t5_input"] = t5_input
            data["t5_answer"] = normalize_t5_input(data["text_answer"])
            data["t5_label"] = normalize_t5_input(["(A)", "(B)"][data["label"]])
        else:
            if self.use_augment:
                segments = [
                    [", ".join(self.augment_contexts.get(data["id"], []))]
                    * len(data["choices"]),
                    [
                        self.normalize_question(data["text_question"]) + " " + ch
                        for ch in data["choices"]
                    ],
                ]
                encoded_sentence = self.tokenizer(
                    *segments,
                    truncation="only_first",
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
            else:
                segments = [
                    [self.normalize_question(data["text_question"])]
                    * len(data["choices"]),
                    data["choices"],
                ]
                encoded_sentence = self.tokenizer(
                    *segments,
                    truncation="only_first",
                    padding="max_length",
                    max_length=self.max_seq_length,
                    return_tensors="pt",
                )
            data["sentence"] = encoded_sentence.input_ids.unsqueeze(0)
            data["mask"] = encoded_sentence.attention_mask.unsqueeze(0)
            data["type_ids"] = encoded_sentence.token_type_ids.unsqueeze(0)
            data["bert_input"] = [
                [segments[0][i], segments[1][i]] for i in range(len(segments[0]))
            ]
            data["bert_label"] = data["label"]
        return data

    def normalize_question(self, question):
        if question.endswith("?") or question.endswith("."):
            return question.capitalize()
        else:
            return (question + ".").capitalize()
