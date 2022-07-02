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
from encoder.dataset.matcher.arc import ARCMatcher
from encoder.utils.settings import (
    preprocess_cache_dir,
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.file import open_file_with_create_directories
from .base import StaticIterableDataset
from .download import ARC, OpenBookQA


class ARCDatasetParallelContext:
    dataset: "ARCDataset" = None


class ARCDataset:
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

        self.matcher = ARCMatcher(tokenizer=self.matcher_tokenizer)
        self.annotator = Annotator()
        self.arc = ARC().require()
        self.openbook_qa = OpenBookQA().require()

        archive_path = os.path.join(preprocess_cache_dir, "arc.data")
        if not os.path.exists(archive_path):
            self.train_data = (
                self.parse_data(self.arc.train_challenge_path, "train_challenge")
                + self.parse_data(self.arc.train_easy_path, "train_easy")
                + self.parse_data(self.arc.validate_easy_path, "validate_easy")
                + self.parse_data(self.arc.test_easy_path, "test_easy")
                # + self.parse_openbook_qa_data(self.openbook_qa.train_path)
                # + self.parse_openbook_qa_data(self.openbook_qa.validate_path)
                # + self.parse_openbook_qa_data(self.openbook_qa.test_path)
            )
            self.validate_data = self.parse_data(
                self.arc.validate_challenge_path, "validate_challenge"
            )
            self.test_data = self.parse_data(
                self.arc.test_challenge_path, "test_challenge"
            )
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
                self.train_data = data["train"]
                self.validate_data = data["validate"]
                self.test_data = data["test"]
        self.set_corpus()
        self.add_kinematics_speed_training_data()
        self.add_kinematics_time_training_data()
        self.add_electric_training_data()

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
                    selection = self.matcher.select_paths(
                        match,
                        max_edges=self.matcher_config["question_select_max_edges"],
                        discard_edges_if_rank_below=self.matcher_config[
                            "question_select_discard_edges_if_rank_below"
                        ],
                    )
                    new_question = self.matcher.insert_selection_at_end_preserve_case(
                        data["text_question"], selection
                    )

                    new_choices = []
                    for choice in data["choices"]:
                        match = self.matcher.match_by_node_embedding(
                            choice,
                            target_sentence=target,
                            target_mask=target_mask,
                            seed=self.matcher_seed,
                            max_times=self.matcher_config["choices_match_max_times"],
                            max_depth=self.matcher_config["choices_match_max_depth"],
                            edge_top_k=self.matcher_config["choices_match_edge_top_k"],
                            source_context_range=self.matcher_config[
                                "choices_match_source_context_range"
                            ],
                        )
                        selection = self.matcher.select_paths(
                            match,
                            max_edges=self.matcher_config["choices_select_max_edges"],
                            discard_edges_if_rank_below=self.matcher_config[
                                "choices_select_discard_edges_if_rank_below"
                            ],
                        )

                        new_choices.append(
                            self.matcher.insert_selection_at_end_preserve_case(
                                choice, selection,
                            )
                        )
                elif self.matcher_mode == "none":
                    new_question = data["text_question"]
                    new_choices = data["choices"]
                else:
                    raise ValueError(f"Invalid match mode {self.matcher_mode}")

                sentences, masks, type_ids = [], [], []
                for choice in new_choices:
                    encoded_sentence = self.tokenizer(
                        new_question,
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
        result = self.annotator.annotate(
            data["text_question"], data["choices"], quiet=quiet
        )
        if result is not None:
            return result
        if data["id"].startswith("openbook_qa"):
            return f" \\n (facts) {', '.join(data['facts'])}"
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
                ("train", "arc_train_for_t5.json", train_data, self.train_data),
                ("validate", "arc_validate_for_t5.json", val_data, self.validate_data),
                ("test", "arc_test_for_t5.json", test_data, self.test_data),
                (
                    "validate_original",
                    "arc_validate_original_for_t5.json",
                    val_original_data,
                    self.validate_data,
                ),
                (
                    "test_original",
                    "arc_test_original_for_t5.json",
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
        ARCDatasetParallelContext.dataset = ARCDataset(
            None,
            matcher_mode=matcher_mode,
            matcher_seed=matcher_seed,
            matcher_config=matcher_config,
        )
        with open(os.path.join(preprocess_cache_dir, "arc_targets.json"), "r") as file:
            ARCDatasetParallelContext.dataset.set_search_targets(json.load(file))

    @staticmethod
    def generate_t5_input(args):
        split, index = args
        if split == "train":
            data = ARCDatasetParallelContext.dataset.train_data[index]
        elif split in ("validate", "validate_original"):
            data = ARCDatasetParallelContext.dataset.validate_data[index]
        else:
            data = ARCDatasetParallelContext.dataset.test_data[index]
        annotation = ""
        if "original" not in split:
            annotation = ARCDatasetParallelContext.dataset.generate_t5_annotation(data)
        return {
            "inputs": ARCDataset.normalize_t5_input(
                data["text_question"]
                + " \\n "
                + data["text_choices"].replace("\n", " ")
                + annotation
            ),
            "targets": ARCDataset.normalize_t5_input(data["text_answer"]),
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
        logits = logits.cpu() - batch["choice_mask"] * 1e7
        score = t.sigmoid(logits).numpy()
        labels = np.argmax(logits.numpy(), axis=1)
        ref_labels = batch["label"].cpu().numpy()

        logit_error_type_count = [0, 0, 0, 0, 0, 0]
        logit_correct_type_count = [0, 0, 0, 0, 0, 0]

        for i in range(len(labels)):
            answer = labels[i]
            ref_answer = ref_labels[i]
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
        labels = self.generate_labels()
        for i in range(tokens.shape[0]):
            answer = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
            ref_answer = self.normalize_t5_input(batch["text_answer"][i])
            sentence = self.tokenizer.decode(
                batch["sentence"][i], skip_special_tokens=True
            )
            answers[batch["id"][i]] = False
            if answer == ref_answer:
                correct += 1
                answers[batch["id"][i]] = True
            elif answer not in labels:
                if self.match_closest_when_no_equal:
                    # Gestalt Pattern Matching
                    # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                    possible_matches = difflib.get_close_matches(
                        answer,
                        [self.normalize_t5_input(c) for c in batch["choices"][i]],
                        n=1,
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
        choice_mask = t.cat([d["choice_mask"] for d in self.test_data], dim=0)
        logits = (logits.cpu() - choice_mask * 1e7).numpy()
        labels = np.argmax(logits, axis=1).tolist()
        with open_file_with_create_directories(
            os.path.join(directory, "arc.csv"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            for label, preprocessed in zip(labels, self.test_data):
                file.write(
                    f"{preprocessed['id']},{preprocessed['choice_labels'][label]}\n"
                )

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "arc.csv"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                choices = [self.normalize_t5_input(c) for c in preprocessed["choices"]]
                for i, choice in enumerate(choices):
                    if answer == choice:
                        file.write(
                            f"{preprocessed['id']},{preprocessed['choice_labels'][i]}\n"
                        )
                        break
                else:
                    is_missing = True
                    if self.match_closest_when_no_equal:
                        # Gestalt Pattern Matching
                        # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                        possible_matches = difflib.get_close_matches(
                            answer, choices, n=1
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
                                f"{preprocessed['choice_labels'][choices.index(possible_matches[0])]}\n"
                            )
                    if is_missing:
                        missing += 1
                        print(
                            f"Missing answer, answer: {answer}, using first label as answer."
                        )
                    file.write(
                        f"{preprocessed['id']},{preprocessed['choice_labels'][0]}"
                    )
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
        wnl = WordNetLemmatizer()
        for split_data in (self.train_data, self.validate_data, self.test_data):
            for data in split_data:
                if data["id"].startswith("generated"):
                    continue
                if len(data["original_split"]) == 0:
                    continue
                key = f"arc_{data['original_split']}_{data['original_index']}"
                if key not in search_target:
                    raise ValueError(f"Entry {key} not found in search data")

                allowed_tokens = []
                # for sentence in search_target[key][:1] + [data["text_question"]]:
                for sentence in [data["text_question"]]:
                    tokens = nltk.word_tokenize(sentence)
                    tagged = nltk.pos_tag(tokens)
                    for token, pos in tagged:
                        if pos.startswith("NN"):
                            allowed_tokens.append(wnl.lemmatize(token.lower()))

                data["target"] = allowed_tokens if len(allowed_tokens) > 0 else [""]
                data["facts"] = search_target[key]

    def parse_data(self, path, original_split):
        data = []
        logging.info(f"Parsing {path}")
        with open_file_with_create_directories(path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                choices = [ch["text"] for ch in entry["question"]["choices"]]
                choice_labels = [ch["label"] for ch in entry["question"]["choices"]]
                original_choice_num = len(choices)
                if len(choices) < 5:
                    diff = 5 - original_choice_num
                    choices += [""] * diff
                    choice_labels += [""] * diff
                choice_mask = t.FloatTensor(
                    [[0 if len(ch) > 0 else 1 for ch in choices]]
                )
                preprocessed = {
                    "text_question": entry["question"]["stem"],
                    "text_choices": self.generate_choice_str(choices),
                    "target": [],
                    "facts": [],
                    "choices": choices,
                    "choice_labels": choice_labels,
                    "choice_mask": choice_mask,
                    "choice_match_masks": self.generate_choice_match_mask(
                        choices[:original_choice_num]
                    )
                    + [""] * (5 - original_choice_num),
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

    def parse_openbook_qa_data(self, path):
        data = []
        logging.info(f"Parsing {path}")
        wnl = WordNetLemmatizer()
        with open_file_with_create_directories(path, "r") as file:
            for line in file:
                entry = json.loads(line)

                choices = [ch["text"] for ch in entry["question"]["choices"]]

                text_choices = self.generate_choice_str(choices + [""])

                tokens = nltk.word_tokenize(entry["fact1"])
                allowed_tokens = []
                tagged = nltk.pos_tag(tokens)
                for token, pos in tagged:
                    if pos.startswith("NN"):
                        allowed_tokens.append(wnl.lemmatize(token.lower()))

                if len(allowed_tokens) < 3:
                    for token, pos in tagged:
                        if pos.startswith("JJ"):
                            allowed_tokens.append(wnl.lemmatize(token.lower()))
                if len(allowed_tokens) == 0:
                    allowed_tokens.append("")

                preprocessed = {
                    "text_question": entry["question"]["stem"] + "?",
                    "text_choices": text_choices,
                    "target": allowed_tokens,
                    "facts": [entry["fact1"]],
                    "choices": choices + [""],
                    "choice_labels": ["A", "B", "C", "D", ""],
                    "choice_mask": t.FloatTensor([[0, 0, 0, 0, 1]]),
                    "choice_match_masks": self.generate_choice_match_mask(choices[:4])
                    + [""],
                    "id": "openbook_qa_" + entry["id"],
                    "original_split": "",
                    "original_index": -1,
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

    def add_kinematics_speed_training_data(self):
        generator = random.Random(42)
        templates = [
            "A {vehicle} takes {time} to travel from city A to city B "
            "which are {distance} apart from each other. "
            "What is the average speed of the {vehicle} during this time?",
            "It takes {time} for a {vehicle} to travel {distance}. "
            "Which best describes the average speed of the {vehicle}?",
            "A passenger boarded a {vehicle} in city A and it takes {time} for her "
            "to get to city B which is {distance} away from city A. "
            "What is the average speed?",
            "The fastest way to travel from city A to city B, where the distance in between "
            "is {distance}, is taking a {vehicle}, and it takes {time} to complete the trip, "
            "What is the average speed of the {vehicle}?",
            "To travel from city A to city B, a {vehicle} takes {time} to cross {distance},"
            "What was the average speed?",
        ]
        for i in range(100):
            vehicle, time, distance = generator.choice(
                [
                    ("car", "1 h", "100 km"),
                    ("car", "2 h", "258 km"),
                    ("car", "7 h", "840 km"),
                    ("car", "1.2 h", "144 km"),
                    ("car", "1.5 h", "181 km"),
                    ("car", "4 h", "470 km"),
                    ("car", "5.2 h", "604 km"),
                    ("car", "3.1 h", "393 km"),
                    ("car", "7 h", "777 km"),
                    ("bus", "3 h", "240 km"),
                    ("bus", "2.4 h", "480 km"),
                    ("bus", "6 h", "540 km"),
                    ("bus", "5 h", "375 km"),
                    ("bus", "2 h", "228 km"),
                    ("bus", "6 h", "528 km"),
                    ("bus", "5 h", "405 km"),
                    ("plane", "2 h", "780 km"),
                    ("plane", "4 h", "840 km"),
                    ("plane", "3.2 h", "1280 km"),
                    ("plane", "5.6 h", "1520 km"),
                    ("plane", "1.2 h", "482 km"),
                    ("plane", "6 h", "2059 km"),
                ]
            )
            question = generator.choice(templates).format(
                vehicle=vehicle, time=time, distance=distance
            )
            answer = float(distance.split(" ")[0]) / float(time.split(" ")[0])
            choices = [
                f"{answer:.2f} km/h",
                f"{answer * 2:.2f} km/h",
                f"{answer * 4:.2f} km/h",
                f"{answer / 2:.2f} km/h",
            ]
            generator.shuffle(choices)
            choices = choices + [""]
            preprocessed = {
                "text_question": question,
                "text_choices": self.generate_choice_str(choices),
                "target": ["speed"],
                "facts": [],
                "choices": choices,
                "choice_labels": ["A", "B", "C", "D", ""],
                "choice_mask": [0, 0, 0, 0, 1],
                "choice_match_masks": self.generate_choice_match_mask(choices[:4])
                + [""],
                "id": "generated_speed_{i}",
                "original_split": "generated_speed",
                "original_index": i,
            }
            label = choices.index(f"{answer:.2f} km/h")
            preprocessed["label"] = label
            preprocessed["text_answer"] = choices[label]
            self.train_data.append(preprocessed)

    def add_kinematics_time_training_data(self):
        generator = random.Random(42)
        templates = [
            "A {vehicle} travels from city A to city B at a speed of {speed}, "
            "and city A is {distance} from B. "
            "How long did it take the {vehicle} to complete the trip?",
            "It is measured that a {vehicle} can travel at {speed}. "
            "How much time does it take for a {vehicle} to travel {distance}?",
            "A passenger boarded a {vehicle} in city A and travels at {speed} "
            "to get to city B which is {distance} away from city A. "
            "How long is the whole trip?",
            "The fastest way to travel from city A to city B, where the distance in between "
            "is {distance}, is taking a {vehicle} which travels at {speed}, "
            "How much time does the trip take?",
        ]
        for i in range(20):
            vehicle, speed, distance, correct_time, wrong_times = generator.choice(
                [
                    (
                        "car",
                        "100 km/h",
                        "100 km",
                        "1 hour",
                        ("12 minutes", "30 minutes", "2 hours"),
                    ),
                    (
                        "car",
                        "130 km/h",
                        "182 km",
                        "1.4 hours",
                        ("35 minutes", "1 hour", "3 hours"),
                    ),
                    (
                        "car",
                        "120 km/h",
                        "840 km",
                        "7 hours",
                        ("3 hours", "6 hours", "9 hours"),
                    ),
                    (
                        "car",
                        "120 km/h",
                        "144 km",
                        "1.2 hours",
                        ("40 minutes", "2 hours", "3 hours"),
                    ),
                    (
                        "car",
                        "60 km/h",
                        "30 km",
                        "30 minutes",
                        ("20 minutes", "45 minutes", "1 hour"),
                    ),
                    (
                        "bus",
                        "80 km/h",
                        "20 km",
                        "15 minutes",
                        ("6 minutes", "20 minutes", "40 minutes"),
                    ),
                    (
                        "bus",
                        "70 km/h",
                        "35 km",
                        "30 minutes",
                        ("25 minutes", "40 minutes", "1 hour"),
                    ),
                    (
                        "plane",
                        "390 km/h",
                        "780 km",
                        "2 hours",
                        ("1 hour", "2.5 hours", "3 hours"),
                    ),
                ]
            )
            question = generator.choice(templates).format(
                vehicle=vehicle, speed=speed, distance=distance
            )
            choices = [correct_time] + list(wrong_times)
            generator.shuffle(choices)
            choices = choices + [""]
            preprocessed = {
                "text_question": question,
                "text_choices": self.generate_choice_str(choices),
                "target": ["time"],
                "facts": [],
                "choices": choices,
                "choice_labels": ["A", "B", "C", "D", ""],
                "choice_mask": [0, 0, 0, 0, 1],
                "choice_match_masks": self.generate_choice_match_mask(choices[:4])
                + [""],
                "id": "generated_time_{i}",
                "original_split": "generated_time",
                "original_index": i,
            }
            label = choices.index(correct_time)
            preprocessed["label"] = label
            preprocessed["text_answer"] = choices[label]
            self.train_data.append(preprocessed)

    def add_electric_training_data(self):
        generator = random.Random(42)
        template = (
            "When there is a current of {amp} amps and a total resistance of {ohm} ohms, "
            "what's the voltage?"
        )
        for i, (amp, ohm, volt) in enumerate(
            ((3, 10, 30), (2, 5, 10), (4, 9, 36), (8, 8, 64), (7, 8, 56))
        ):

            question = template.format(amp=amp, ohm=ohm)
            choices = [
                f"{volt:.2f} V",
                f"{volt * 2:.2f} V",
                f"{volt * 4:.2f} V",
                f"{volt / 2:.2f} V",
            ]
            generator.shuffle(choices)
            choices = choices + [""]
            preprocessed = {
                "text_question": question,
                "text_choices": self.generate_choice_str(choices),
                "target": ["time"],
                "facts": [],
                "choices": choices,
                "choice_labels": ["A", "B", "C", "D", ""],
                "choice_mask": [0, 0, 0, 0, 1],
                "choice_match_masks": self.generate_choice_match_mask(choices[:4])
                + [""],
                "id": "generated_electric_{i}",
                "original_split": "generated_electric",
                "original_index": i,
            }
            label = choices.index(f"{volt:.2f} V")
            preprocessed["label"] = label
            preprocessed["text_answer"] = choices[label]
            self.train_data.append(preprocessed)

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

    def __reduce__(self):
        return (
            ARCDataset,
            (
                self.tokenizer,
                self.max_seq_length,
                self.use_matcher,
                self.matcher_mode,
                self.matcher_seed,
                self.matcher_config,
                self.match_closest_when_no_equal,
                self.output_mode,
            ),
        )
