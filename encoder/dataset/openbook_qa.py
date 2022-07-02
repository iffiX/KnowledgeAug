import os
import re
import copy
import json
import tqdm
import pickle
import random
import difflib
import logging
import nltk
import multiprocessing
import numpy as np
import torch as t
from typing import List, Dict
from nltk.stem import WordNetLemmatizer
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
from encoder.utils.file import open_file_with_create_directories
from .utils import num2word
from .base import StaticIterableDataset


class OpenBookQADatasetParallelContext:
    dataset: "OpenBookQADataset" = None


class OpenBookQADataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 300,
        use_matcher: bool = False,
        matcher_mode: str = "embedding",
        matcher_seed: int = -1,
        matcher_config: dict = None,
        match_closest_when_no_equal: bool = True,
        output_mode: str = "single",
    ):
        self.tokenizer = tokenizer
        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.max_seq_length = max_seq_length
        self.use_matcher = use_matcher
        self.matcher_mode = matcher_mode
        self.matcher_seed = matcher_seed
        self.matcher_config = matcher_config
        self.match_closest_when_no_equal = match_closest_when_no_equal
        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")
        self.output_mode = output_mode

        self.matcher = OpenBookQAMatcher(tokenizer=self.matcher_tokenizer)
        self.openbook_qa = OpenBookQA().require()

        archive_path = os.path.join(preprocess_cache_dir, "openbook_qa.data")
        if not os.path.exists(archive_path):
            self.train_data = self.parse_data(self.openbook_qa.train_path, "train")
            self.validate_data = self.parse_data(
                self.openbook_qa.validate_path, "validate"
            )
            self.test_data = self.parse_data(self.openbook_qa.test_path, "test")
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
                self.train_data = data["train"]
                self.validate_data = data["validate"]
                self.test_data = data["test"]

        self.original_train_data = copy.deepcopy(self.train_data)
        self.original_validate_data = copy.deepcopy(self.validate_data)
        self.original_test_data = copy.deepcopy(self.test_data)
        self.set_corpus()

        if output_mode == "splitted":
            self.normalize_training_data()

        # self.add_arithmetic_training_data()

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
        if split == "train":
            data = self.train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        else:
            data = self.test_data[index]
        if self.output_mode == "single":
            if self.use_matcher:
                annotation = self.generate_t5_annotation(data, quiet=True)
                encoded_sentence = self.tokenizer(
                    self.normalize_t5_input(
                        data["text_question"]
                        + " \\n "
                        + data["text_choices"].replace("\n", " ")
                        + annotation
                    ),
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            else:
                encoded_sentence = self.tokenizer(
                    self.normalize_t5_input(
                        data["text_question"]
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
            if self.use_matcher:
                # prevent any modification to data, also prevent checkpoint storing
                # data to gpu by moving
                data = copy.deepcopy(data)
                if self.matcher_mode == "embedding":
                    target = " @ ".join(sorted(list(set(data["target"]))))
                    target_mask = []
                    for c in target:
                        if c == "@":
                            target_mask.append("-")
                        else:
                            target_mask.append("+")
                    target_mask = "".join(target_mask)

                    match = self.matcher.match_by_node_embedding(
                        data["text_question"],
                        target_sentence=target,
                        target_mask=target_mask,
                        seed=self.matcher_seed,
                        max_times=self.matcher_config["question_match_max_times"],
                        max_depth=self.matcher_config["question_match_max_depth"],
                        edge_top_k=self.matcher_config["question_match_edge_top_k"],
                        source_context_range=self.matcher_config[
                            "question_match_source_context_range"
                        ],
                    )
                    question_selection = self.matcher.select_paths(
                        match,
                        max_edges=self.matcher_config["question_select_max_edges"],
                        discard_edges_if_rank_below=self.matcher_config[
                            "question_select_discard_edges_if_rank_below"
                        ],
                    )
                    # new_question = (
                    #     self.matcher.insert_selection_at_end_preserve_case(
                    #         data["text_question"], selection
                    #     )
                    #     + " (facts) "
                    #     + ", ".join(data["facts"])
                    # )

                    new_questions = []
                    new_choices = []
                    for choice, match_mask in zip(
                        data["choices"],
                        self.generate_choice_match_mask(data["choices"]),
                    ):
                        match = self.matcher.match_by_node_embedding(
                            choice,
                            target_sentence=target,
                            source_mask=match_mask,
                            target_mask=target_mask,
                            seed=self.matcher_seed,
                            max_times=self.matcher_config["choices_match_max_times"],
                            max_depth=self.matcher_config["choices_match_max_depth"],
                            edge_top_k=self.matcher_config["choices_match_edge_top_k"],
                            source_context_range=self.matcher_config[
                                "choices_match_source_context_range"
                            ],
                        )
                        choice_selection = self.matcher.select_paths(
                            match,
                            max_edges=self.matcher_config["choices_select_max_edges"],
                            discard_edges_if_rank_below=self.matcher_config[
                                "choices_select_discard_edges_if_rank_below"
                            ],
                        )

                        # new_choices.append(
                        #     self.matcher.insert_selection_at_end_preserve_case(
                        #         data["text_question"] + " " + choice, selection,
                        #     )
                        # )
                        new_choices.append(data["text_question"] + " " + choice)
                        new_questions.append(
                            self.matcher.insert_selection_at_end_preserve_case(
                                self.matcher.insert_selection_at_end_preserve_case(
                                    ", ".join(data["facts"]),
                                    question_selection,
                                    begin="",
                                    end="",
                                ),
                                choice_selection,
                                begin="",
                                end="",
                            )
                        )
                    # new_questions = [", ".join(data["facts"])] * len(data["choices"])
                elif self.matcher_mode == "none":
                    new_questions = [", ".join(data["facts"])] * len(data["choices"])
                    new_choices = [
                        data["text_question"] + " " + ch for ch in data["choices"]
                    ]
                else:
                    raise ValueError(f"Invalid match mode {self.matcher_mode}")

                sentences, masks, type_ids = [], [], []
                for question, choice in zip(new_questions, new_choices):
                    encoded_sentence = self.tokenizer(
                        question,
                        choice,
                        padding="max_length",
                        max_length=self.max_seq_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    sentences.append(encoded_sentence.input_ids)
                    masks.append(encoded_sentence.attention_mask)
                    type_ids.append(encoded_sentence.token_type_ids)
                data["sentence"] = t.stack(sentences, dim=1)
                data["mask"] = t.stack(masks, dim=1)
                data["type_ids"] = t.stack(type_ids, dim=1)
            else:
                sentences, masks, type_ids = [], [], []
                for choice in data["choices"]:
                    encoded_sentence = self.tokenizer(
                        data["text_question"],
                        choice,
                        padding="max_length",
                        max_length=self.max_seq_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    sentences.append(encoded_sentence.input_ids)
                    masks.append(encoded_sentence.attention_mask)
                    type_ids.append(encoded_sentence.token_type_ids)
                data["sentence"] = t.stack(sentences, dim=1)
                data["mask"] = t.stack(masks, dim=1)
                data["type_ids"] = t.stack(type_ids, dim=1)
        return data

    def generate_t5_annotation(self, data, quiet=False):
        # prevent any modification to data, also prevent checkpoint storing
        # data to gpu by moving
        data = copy.deepcopy(data)
        # result = self.annotator.annotate(
        #     data["text_question"], data["choices"], quiet=quiet
        # )
        # if result is not None:
        #     return result
        if self.matcher_mode == "embedding":
            if len(data["target"]) == 0:
                raise ValueError(f"Target not set for data with id {data['id']}")
            target = " @ ".join(sorted(list(set(data["target"]))))
            target_mask = []
            for c in target:
                if c == "@":
                    target_mask.append("-")
                else:
                    target_mask.append("+")
            target_mask = "".join(target_mask)

            fact_annotation = f"(facts) {', '.join(data['facts'])}"

            match = self.matcher.match_by_node_embedding(
                data["text_question"],
                target_sentence=target,
                target_mask=target_mask,
                seed=self.matcher_seed,
                max_times=self.matcher_config["question_match_max_times"],
                max_depth=self.matcher_config["question_match_max_depth"],
                edge_top_k=self.matcher_config["question_match_edge_top_k"],
                source_context_range=self.matcher_config[
                    "question_match_source_context_range"
                ],
                source_context_weight=1,
            )
            selection = self.matcher.select_paths(
                match,
                max_edges=self.matcher_config["question_select_max_edges"],
                discard_edges_if_rank_below=self.matcher_config[
                    "question_select_discard_edges_if_rank_below"
                ],
            )

            question_annotation = self.matcher.insert_selection_at_end_preserve_case(
                "(question) ", selection, begin="", end="",
            )

            choice_annotations = []
            for label, choice, match_mask in zip(
                self.generate_labels(),
                data["choices"],
                # self.generate_choice_match_mask(data["choices"])
                data["choice_match_masks"],
            ):
                if len(choice) > 0:
                    match = self.matcher.match_by_node_embedding(
                        choice,
                        target_sentence=target,
                        source_mask=match_mask,
                        target_mask=target_mask,
                        seed=self.matcher_seed,
                        max_times=self.matcher_config["choices_match_max_times"],
                        max_depth=self.matcher_config["choices_match_max_depth"],
                        edge_top_k=self.matcher_config["choices_match_edge_top_k"],
                        source_context_range=self.matcher_config[
                            "choices_match_source_context_range"
                        ],
                        source_context_weight=1,
                    )
                    selection = self.matcher.select_paths(
                        match,
                        max_edges=self.matcher_config["choices_select_max_edges"],
                        discard_edges_if_rank_below=self.matcher_config[
                            "choices_select_discard_edges_if_rank_below"
                        ],
                    )

                    choice_annotations.append(
                        self.matcher.insert_selection_at_end_preserve_case(
                            f"({choice}) ", selection, begin="", end=""
                        )
                    )
            annotation = (
                fact_annotation
                + " "
                + question_annotation
                + " "
                + " ".join(choice_annotations)
            )
        elif self.matcher_mode == "none":
            annotation = ""
        else:
            raise ValueError(f"Invalid match mode {self.matcher_mode}")

        annotation = " \\n " + annotation if len(annotation) > 0 else annotation
        return annotation

    def generate_all_t5_data(self, split=None):
        train_data, val_data, test_data, val_original_data, test_original_data = (
            [],
            [],
            [],
            [],
            [],
        )
        with multiprocessing.Pool(
            initializer=self.initialize_pool,
            initargs=(self.matcher_mode, self.matcher_seed, self.matcher_config),
        ) as pool:
            for process_split, path, target, source in (
                ("train", "openbook_qa_train_for_t5.json", train_data, self.train_data),
                (
                    "validate",
                    "openbook_qa_validate_for_t5.json",
                    val_data,
                    self.validate_data,
                ),
                ("test", "openbook_qa_test_for_t5.json", test_data, self.test_data),
                (
                    "validate_original",
                    "openbook_qa_validate_original_for_t5.json",
                    val_original_data,
                    self.validate_data,
                ),
                (
                    "test_original",
                    "openbook_qa_test_original_for_t5.json",
                    test_original_data,
                    self.test_data,
                ),
            ):
                if split is not None:
                    if (isinstance(split, str) and process_split != split) or (
                        isinstance(split, list) and process_split not in split
                    ):
                        continue
                print(f"Processing {process_split}")
                with tqdm.tqdm(total=len(source)) as pbar:
                    for result in pool.imap_unordered(
                        self.generate_t5_input,
                        [(process_split, i) for i in range(len(source))],
                    ):
                        pbar.update()
                        target.append(result)

                with open(os.path.join(preprocess_cache_dir, path), "w") as file:
                    json.dump(target, file, indent=2)

    @staticmethod
    def initialize_pool(matcher_mode, matcher_seed, matcher_config):
        OpenBookQADatasetParallelContext.dataset = OpenBookQADataset(
            None,
            matcher_mode=matcher_mode,
            matcher_seed=matcher_seed,
            matcher_config=matcher_config,
        )
        with open(
            os.path.join(preprocess_cache_dir, "openbook_qa_targets.json"), "r"
        ) as file:
            OpenBookQADatasetParallelContext.dataset.set_search_targets(json.load(file))

    @staticmethod
    def generate_t5_input(args):
        split, index = args
        if split == "train":
            data = OpenBookQADatasetParallelContext.dataset.train_data[index]
        elif split in ("validate", "validate_original"):
            data = OpenBookQADatasetParallelContext.dataset.validate_data[index]
        else:
            data = OpenBookQADatasetParallelContext.dataset.test_data[index]
        annotation = ""
        if "original" not in split:
            annotation = OpenBookQADatasetParallelContext.dataset.generate_t5_annotation(
                data
            )
        return {
            "inputs": OpenBookQADataset.normalize_t5_input(
                data["text_question"]
                + " \\n "
                + data["text_choices"].replace("\n", " ")
                + annotation
            ),
            "targets": OpenBookQADataset.normalize_t5_input(data["text_answer"]),
            "choices": data["choices"],
            "label": data["label"],
            "id": data["id"],
        }

    @staticmethod
    def normalize_t5_input(text):
        text = text.lower()
        text = re.sub(r"'(.*)'", r"\1", text)
        return text

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

    def set_search_targets(self, search_target: Dict[str, List[str]]):
        for split_data in (self.validate_data, self.test_data):
            for data in split_data:
                if data["id"].startswith("generated"):
                    continue
                if len(data["original_split"]) == 0:
                    continue
                key = f"openbook_qa_{data['original_split']}_{data['original_index']}"
                if key not in search_target:
                    raise ValueError(f"Entry {key} not found in search data")

                allowed_tokens = []
                # currently use top 1 as search target, can be set to more
                for sentence in search_target[key][:1]:
                    allowed_tokens += self.extract_targets(sentence)

                data["target"] = allowed_tokens if len(allowed_tokens) > 0 else [""]
                data["facts"] = search_target[key]

    def parse_data(self, path, original_split):
        data = []
        logging.info(f"Parsing {path}")
        with open_file_with_create_directories(path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                text_choices = self.generate_choice_str(
                    [ch["text"] for ch in entry["question"]["choices"]]
                )

                choices = [
                    f"{ch['text'].lower().strip(',')}"
                    for ch in entry["question"]["choices"]
                ]

                preprocessed = {
                    "text_question": entry["question"]["stem"].lower() + "?",
                    "text_choices": text_choices,
                    "target": self.extract_targets(entry["fact1"])
                    if original_split == "train"
                    else [],
                    "original_facts": [entry["fact1"]],
                    "facts": [entry["fact1"]] if original_split == "train" else [],
                    "choices": choices,
                    "choice_match_masks": self.generate_choice_match_mask(choices),
                    "id": entry["id"],
                    "original_split": original_split,
                    "original_index": idx,
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

    def extract_targets(self, sentence):
        wnl = WordNetLemmatizer()
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        allowed_tokens = []
        for token, pos in tagged:
            if pos.startswith("NN"):
                allowed_tokens.append(wnl.lemmatize(token.lower()))
        return allowed_tokens

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "train": self.train_data,
                    "validate": self.validate_data,
                    "test": self.test_data,
                },
                file,
            )

    def add_arithmetic_training_data(self):
        generator = random.Random(42)
        # TIME ARITHMETIC
        question_template = "{month_1} is {time_diff} {direction}?"
        answer_template = "{month_2}"
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        for i in range(100):
            month_1, month_2 = generator.choices(list(range(12)), k=2)
            direction = generator.choice(["before", "after"])
            if direction == "before":
                time_diff = (month_2 + 12 - month_1) % 12
            else:
                time_diff = (month_1 + 12 - month_2) % 12
            unit, unit_scale = generator.choice(
                # [("month", 1), ("week", 4), ("day", 30)]
                [("month", 1)]
            )
            wrong_months = months[:month_2] + months[month_2 + 1 :]
            generator.shuffle(wrong_months)
            choices = [
                answer_template.format(month_2=months[month_2]),
                answer_template.format(month_2=wrong_months[0]),
                answer_template.format(month_2=wrong_months[1]),
                answer_template.format(month_2=wrong_months[2]),
            ]
            sample = {
                "text_question": question_template.format(
                    month_1=months[month_1],
                    time_diff=f"{num2word(time_diff * unit_scale)} "
                    f"{unit}{'s' if time_diff * unit_scale > 1 else ''}",
                    direction=direction,
                ),
                "text_choices": self.generate_choice_str(choices),
                "fact": "month week day",
                "target": [],
                "choices": choices,
                "id": f"gen-ar-time-{i}",
                "label": 0,
                "text_answer": choices[0],
            }
            self.train_data.append(sample)
        logging.info("Added 100 samples of time arithmetic")

        # TIME CONVERSION
        time_units = ["day", "week", "month", "season", "year"]
        time_unit_names = {
            "day": ["day"],  # ["day", "24-hour period", "working day", "solar day"],
            "week": ["week"],  # ["week", "weekend"],
            "month": [
                "month"
            ],  # ["month", "calendar page", "full moon phase", "lunar month"],
            "season": ["season"],  # ["season", "3-month period", "quarter of year"],
            "year": ["year"],
        }
        time_unit_conversion = [
            [-1, 7, 30, 90, 365],
            [-1, -1, 4, 12, 52],
            [-1, -1, -1, 3, 12],
            [-1, -1, -1, -1, 4],
            [-1, -1, -1, -1, -1],
        ]
        question_template = (
            "How many {time_unit_1}s are there in {time_number_2} {time_unit_2}?"
        )
        answer_template = "{time_number_1}"
        for i in range(100):
            time_unit_1 = generator.choice(time_units[:4])
            time_unit_1_name = generator.choice(time_unit_names[time_unit_1])
            time_unit_2 = generator.choice(
                time_units[time_units.index(time_unit_1) + 1 :]
            )
            time_unit_2_name = generator.choice(time_unit_names[time_unit_2])

            time_number_2 = generator.randint(1, 3)
            time_number_1 = (
                time_number_2
                * time_unit_conversion[time_units.index(time_unit_1)][
                    time_units.index(time_unit_2)
                ]
            )
            wrong_numbers = list(
                set(range(1, time_number_1 * 2)).difference({time_number_1})
            )
            generator.shuffle(wrong_numbers)
            choices = [
                answer_template.format(time_number_1=num2word(time_number_1)),
                answer_template.format(time_number_1=num2word(wrong_numbers[0])),
                answer_template.format(time_number_1=num2word(wrong_numbers[1])),
                answer_template.format(time_number_1=num2word(wrong_numbers[2])),
            ]
            sample = {
                "text_question": question_template.format(
                    time_unit_1=time_unit_1_name,
                    time_number_2=num2word(time_number_2),
                    time_unit_2=f"{time_unit_2_name}{'s' if time_number_2 > 1 else ''}",
                ),
                "text_choices": self.generate_choice_str(choices),
                "fact": "month week day",
                "target": [],
                "choices": choices,
                "id": f"gen-ar-timeconv-{i}",
                "label": 0,
                "text_answer": choices[0],
            }
            self.train_data.append(sample)
        logging.info("Added 100 samples of time conversion")

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

    def generate_choice_match_mask(self, choices: List[str]):
        valid_choice_num = sum(len(choice) > 0 for choice in choices)
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
                ) == valid_choice_num or any(
                    token in other_lemma_tokens_set
                    for other_lemma_tokens_set in choices_lemma_tokens_set[:choice_idx]
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
