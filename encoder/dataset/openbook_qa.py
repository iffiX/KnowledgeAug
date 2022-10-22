import os
import re
import copy
import json
import nltk
import random
import difflib
import logging
import numpy as np
import torch as t
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple, Dict, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.download import OpenBookQA
from encoder.dataset.matcher.openbook_qa import OpenBookQAMatcher
from encoder.utils.settings import (
    preprocess_cache_dir,
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.file import PickleCache, open_file_with_create_directories
from .base import StaticIterableDataset
from .utils import normalize_t5_input


class OpenBookQABaseDataset:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        match_closest_when_no_equal: bool = True,
        output_mode: str = "single",
    ):
        self.tokenizer = tokenizer
        self.match_closest_when_no_equal = match_closest_when_no_equal
        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")
        self.output_mode = output_mode

        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )

        self.matcher = OpenBookQAMatcher(tokenizer=self.matcher_tokenizer)
        self.openbook_qa = OpenBookQA().require()

        with PickleCache(
            os.path.join(preprocess_cache_dir, "openbook_qa.data"), self.generate_data,
        ) as cache:
            self.train_data = cache.data["train"]
            self.validate_data = cache.data["validate"]
            self.test_data = cache.data["test"]

        # since we will normalize the training data, keep a copy of the original data
        self.original_train_data = copy.deepcopy(self.train_data)
        self.validate_openbook_qa_corpus_retrieval_rate()
        self.set_corpus()
        if output_mode == "splitted":
            self.normalize_training_data()

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
        score = t.sigmoid(logits.cpu()).numpy()
        logits = logits.cpu().numpy()
        # labels = np.argmax(logits, axis=1)
        labels = self.apply_logical_ops_to_logits(batch["choices"], logits)
        ref_labels = batch["label"].cpu().numpy()

        logit_error_type_count = [0, 0, 0, 0, 0]
        logit_correct_type_count = [0, 0, 0, 0, 0]

        for i in range(len(labels)):
            answer = ["A", "B", "C", "D"][labels[i]]
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
                print(
                    f"answer: [{answer}] \n"
                    f"ref_answer: [{ref_answer}] \n"
                    f"logits: [{score[i].tolist()}]"
                )
                postive_count = np.sum(score[i] > 0.1)
                logit_error_type_count[postive_count] += 1
            else:
                postive_count = np.sum(score[i] > 0.1)
                logit_correct_type_count[postive_count] += 1
        print(f"Logit error types: {logit_error_type_count}")
        print(f"Logit correct types: {logit_correct_type_count}")
        return {"accuracy": float(np.sum(labels == ref_labels)) / len(labels)}

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
            os.path.join(directory, "openbook_qa.csv"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(f"{preprocessed['id']},{answer_keys[label]}\n")

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "openbook_qa.csv"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(preprocessed["choices"]):
                    if answer == choice:
                        file.write(f"{preprocessed['id']},{answer_keys[i]}\n")
                        break
                else:
                    is_missing = True
                    if self.match_closest_when_no_equal:
                        # Gestalt Pattern Matching
                        # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                        possible_matches = difflib.get_close_matches(
                            answer, preprocessed["choices"], n=1
                        )
                        if not len(possible_matches) == 0:
                            print(
                                f"Using answer {possible_matches[0]} for output {answer}, "
                                f"question: {preprocessed['text_question']}, "
                                f"choices: {preprocessed['choices']}"
                            )
                            is_missing = False
                            file.write(
                                f"{preprocessed['id']},"
                                f"{answer_keys[preprocessed['choices'].index(possible_matches[0])]}\n"
                            )

                    if is_missing:
                        missing += 1
                        print(
                            f"Missing answer, choices: {preprocessed['choices']}, "
                            f"answer: {answer}, using default A as answer."
                        )
                        file.write(f"{preprocessed['id']},A")
        print(f"Missing ratio {float(missing) / len(self.test_data)}")

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
            "train": self.parse_data(self.openbook_qa.train_path, "train"),
            "validate": self.parse_data(self.openbook_qa.validate_path, "validate"),
            "test": self.parse_data(self.openbook_qa.test_path, "test"),
        }

    def parse_data(self, path, split):
        data = []
        logging.info(f"Parsing {path}")
        with open_file_with_create_directories(path, "r") as file:
            for line in file:
                entry = json.loads(line)
                text_choices = self.generate_choice_str(
                    [ch["text"] for ch in entry["question"]["choices"]]
                )

                choices = [
                    f"{ch['text'].replace(',', '').strip('.').lower()}"
                    for ch in entry["question"]["choices"]
                ]

                preprocessed = {
                    "text_question": entry["question"]["stem"].lower()
                    + ("?" if not entry["question"]["stem"].endswith("?") else ""),
                    "text_choices": text_choices,
                    "original_facts": [self.normalize_fact(entry["fact1"])],
                    "facts": [self.normalize_fact(entry["fact1"])]
                    if split == "train"
                    else [],
                    "choices": choices,
                    "choice_masks": self.generate_choice_match_mask(choices),
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

    def normalize_training_data(self):
        append_train_data = []
        delete_train_data = set()
        data = self.train_data
        available_choices = list(
            set([ch for train_data in data for ch in train_data["choices"]]).difference(
                {"all of these", "none of these"}
            )
        )
        generator = random.Random(42)
        for train_idx, train_data in enumerate(data):
            correct_choice = train_data["choices"][train_data["label"]].lower()
            if correct_choice == "all of these":
                # randomly choose 3 choices from other samples
                choice_num = len(train_data["choices"])
                for choice_idx in range(choice_num):
                    if choice_idx != train_data["label"]:
                        new_train_data = copy.deepcopy(train_data)
                        new_train_data["label"] = choice_idx
                        for other_choice_idx in range(choice_num):
                            if other_choice_idx != choice_idx:
                                new_train_data["choices"][
                                    other_choice_idx
                                ] = generator.choice(available_choices)
                        append_train_data.append(new_train_data)
                delete_train_data.add(train_idx)
            elif correct_choice == "none of these":
                delete_train_data.add(train_idx)
        new_data = [
            train_data
            for train_idx, train_data in enumerate(data)
            if train_idx not in delete_train_data
        ] + append_train_data
        logging.info(
            f"Appended {len(append_train_data)} samples, Deleted {len(delete_train_data)} samples"
        )
        self.train_data = new_data

    @staticmethod
    def apply_logical_ops_to_logits(choices: List[List[str]], logits: np.ndarray):
        labels = []
        for ch, lo in zip(choices, logits):
            ch = [c.lower() for c in ch]
            all_of_index, none_of_index, multiple_index = None, None, None
            multiple_choices = None
            if "all of these" in ch:
                all_of_index = ch.index("all of these")
            if "none of these" in ch:
                none_of_index = ch.index("none of these")
            for idx, c in enumerate(ch):
                match_result = re.match(r"([abcd]) and ([abcd])", c)
                if match_result is not None:
                    multiple_index = idx
                    multiple_choices = [
                        ord(match_result.group(1)) - ord("a"),
                        ord(match_result.group(2)) - ord("a"),
                    ]
            if any(
                [
                    all_of_index is not None,
                    # none_of_index is not None,
                    multiple_index is not None,
                ]
            ):
                normal_choices = list(
                    {0, 1, 2, 3}.difference(
                        {all_of_index, none_of_index, multiple_index}
                    )
                )
                if all_of_index is not None and np.all(lo[normal_choices] > 0.1):
                    labels.append(all_of_index)
                elif none_of_index is not None and np.all(lo[normal_choices] < 0.1):
                    labels.append(none_of_index)
                elif multiple_index is not None and np.all(lo[multiple_choices] > 0.1):
                    labels.append(multiple_index)
                else:
                    labels.append(normal_choices[np.argmax(lo[normal_choices])])
            else:
                labels.append(np.argmax(lo))
        return np.array(labels)

    def validate_openbook_qa_corpus_retrieval_rate(self):
        for split, data in zip(
            ("train", "validate", "test"),
            (self.train_data, self.validate_data, self.test_data),
        ):
            fact_allowed_count = 0
            for d in data:
                if (
                    self.matcher.fact_to_composite_node.get(d["original_facts"][0], -1)
                    in self.matcher.allowed_composite_nodes[d["id"]]
                ):
                    fact_allowed_count += 1
            logging.info(
                f"Allowed facts: {split} retrieved ratio: {fact_allowed_count / len(data)}"
            )

    def generate_choice_match_mask(self, choices: List[str]):
        if all(c.count(" ") < 3 for c in choices):
            choice_num = len(choices)
            wnl = WordNetLemmatizer()
            choices_tokens = [nltk.word_tokenize(choice) for choice in choices]
            choices_lemma_tokens = [
                [wnl.lemmatize(token.lower()) for token in tokens]
                for tokens in choices_tokens
            ]
            choices_lemma_tokens_set = [
                set(lemma_tokens) for lemma_tokens in choices_lemma_tokens
            ]
            choices_token_is_common = [[] for _ in range(len(choices))]
            # find common tokens
            for choice_idx, (lemma_tokens, common_list) in enumerate(
                zip(choices_lemma_tokens, choices_token_is_common)
            ):
                for token in lemma_tokens:
                    if sum(
                        token in other_lemma_tokens_set
                        for other_lemma_tokens_set in choices_lemma_tokens_set
                    ) == choice_num or any(
                        token in other_lemma_tokens_set
                        for other_lemma_tokens_set in choices_lemma_tokens_set[
                            :choice_idx
                        ]
                    ):
                        common_list.append(True)
                    else:
                        common_list.append(False)

            # generate mask
            masks = []
            for choice, tokens, common_list in zip(
                choices, choices_tokens, choices_token_is_common
            ):
                mask = ["+"] * len(choice)
                start = 0
                for token, is_common in zip(tokens, common_list):
                    if is_common and re.search(r"^[a-zA-Z]", token):
                        start = choice.index(token, start)
                        mask[start : start + len(token)] = ["-"] * len(token)
                masks.append("".join(mask))
            return masks
        else:
            # Do not generate the last level path to target nodes for long choices
            return ["-" * len(c) for c in choices]

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

    @staticmethod
    def normalize_fact(fact):
        return fact.strip("\n").strip(".").strip('"').strip("'").strip(",").lower()


class OpenBookQAAugmentDataset(OpenBookQABaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
        max_seq_length: int = 300,
        output_mode: str = "single",
    ):
        self.augment_contexts = augment_contexts
        self.max_seq_length = max_seq_length
        self.rand = random.Random(42)
        self.rand_train = True
        super(OpenBookQAAugmentDataset, self).__init__(
            tokenizer, output_mode=output_mode,
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
                ref_context = self.augment_contexts[1].get(data_id)
                augment_context = self.augment_contexts[0].get(data_id, None)
                if augment_context is not None:
                    augment_context = self.rand.choice([ref_context, augment_context])
                else:
                    augment_context = ref_context
            else:
                augment_context = self.augment_contexts[1].get(data_id)
        return augment_context
