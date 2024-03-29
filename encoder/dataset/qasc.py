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
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    BatchEncoding,
)
from encoder.dataset.download import QASC
from encoder.dataset.matcher.qasc import QASCMatcher
from encoder.models.embedder import Embedder
from encoder.utils.settings import (
    preprocess_cache_dir,
    model_cache_dir,
    local_files_only,
    proxies,
    huggingface_mirror,
)
from encoder.utils.file import JSONCache, PickleCache, open_file_with_create_directories
from .base import StaticIterableDataset
from .utils import normalize_t5_input


class QASCBaseDataset:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        match_closest_when_no_equal: bool = True,
    ):
        self.tokenizer = tokenizer
        self.match_closest_when_no_equal = match_closest_when_no_equal

        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.matcher = QASCMatcher(tokenizer=self.matcher_tokenizer)
        self.qasc = QASC().require()

        with PickleCache(
            os.path.join(preprocess_cache_dir, "qasc.data"), self.generate_data,
        ) as cache:
            self.train_data = cache.data["train"]
            self.validate_data = cache.data["validate"]
            self.test_data = cache.data["test"]

        with JSONCache(
            os.path.join(preprocess_cache_dir, "qasc_relevant_questions.json"),
            self.generate_relevant_question_ids,
        ) as cache:
            self.relevant_questions = cache.data

        # since we will shuffle the training data (for randomly generating the reward predictor dataset),
        # keep a copy of the original data
        self.original_train_data = copy.deepcopy(self.train_data)
        rand = random.Random(42)
        rand.shuffle(self.train_data)
        self.set_corpus()
        self.validate_qasc_corpus_retrieval_rate()
        self.test_reference = self.parse_reference(self.qasc.reference_path)
        for i in range(len(self.test_data)):
            if i < 300:
                self.test_data[i]["label"] = self.test_reference[
                    self.test_data[i]["id"]
                ]
            else:
                self.test_data[i]["label"] = -1

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

    @property
    def test_dataset_with_reference(self):
        return StaticIterableDataset(300, self.generator, ("test",))

    def generator(self, index: int, split: str):
        """
        Needs to be overridden
        """
        raise NotImplementedError()

    def validate_logits(self, batch: BatchEncoding, logits: t.Tensor):
        """
        For use with a classifier model
        """
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1)
        ref_labels = batch["label"].cpu().numpy()

        for i in range(labels.shape[0]):
            answer = ["A", "B", "C", "D", "E", "F", "G", "H"][labels[i]]
            ref_answer = ["A", "B", "C", "D", "E", "F", "G", "H"][batch["label"][i]]

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
        choices = [
            normalize_t5_input(x)
            for x in ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
        ]
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
            os.path.join(directory, "qasc.csv"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E", "F", "G", "H"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(f'"{preprocessed["id"]}","{answer_keys[label]}"\n')

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        choices = [
            normalize_t5_input(x)
            for x in ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"]
        ]
        with open_file_with_create_directories(
            os.path.join(directory, "qasc.csv"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E", "F", "G", "H"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(choices):
                    if answer == choice:
                        file.write(f'"{preprocessed["id"]}","{answer_keys[i]}"\n')
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
                                f"choices: {choices}"
                            )
                            is_missing = False
                            file.write(
                                f'"{preprocessed["id"]}",'
                                f'"{answer_keys[choices.index(possible_matches[0])]}"\n'
                            )
                    if is_missing:
                        missing += 1
                        print(
                            f"Missing answer, choices: {choices}, "
                            f"answer: {answer}, using default A as answer."
                        )
                        file.write(f'"{preprocessed["id"]}","A"\n')
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
            "train": self.sort_facts(self.parse_data(self.qasc.train_path, "train")),
            "validate": self.sort_facts(
                self.parse_data(self.qasc.validate_path, "validate")
            ),
            "test": self.parse_data(self.qasc.test_path, "test"),
        }

    def sort_facts(self, split_data):
        import torch
        import torch.nn.functional as F

        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[
                0
            ]  # First element of model_output contains all token embeddings
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        model = AutoModel.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        ).to(f"cuda:0")
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )

        questions = [d["text_question"] for d in split_data]
        facts_1 = [d["original_facts"][0] for d in split_data]
        facts_2 = [d["original_facts"][1] for d in split_data]
        embeddings = []
        for sentences in (questions, facts_1, facts_2):
            encoded_input = tokenizer(
                sentences, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                model_output = model(**encoded_input.to("cuda:0"))
            sentence_embeddings = mean_pooling(
                model_output, encoded_input["attention_mask"]
            )
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embeddings.append(sentence_embeddings)
        facts_1_similarity = torch.sum(embeddings[0] * embeddings[1], dim=1)
        facts_2_similarity = torch.sum(embeddings[0] * embeddings[2], dim=1)
        facts_1_first = facts_1_similarity > facts_2_similarity
        for data, is_first in zip(split_data, facts_1_first):
            if not is_first:
                data["original_facts"] = [
                    data["original_facts"][1],
                    data["original_facts"][0],
                ]
                if len(data["facts"]) > 0:
                    data["facts"] = [data["facts"][1], data["facts"][0]]
                print(
                    f"Reversed: "
                    f"question: {data['text_question']} ||| "
                    f"new_fact1: {data['original_facts'][0]} ||| "
                    f"new_fact2: {data['original_facts'][1]}"
                )
        return split_data

    def parse_data(self, path, split):
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
                    "original_facts": [
                        self.normalize_fact(entry["fact1"]),
                        self.normalize_fact(entry["fact2"]),
                    ]
                    if split != "test"
                    else [],
                    "facts": [
                        self.normalize_fact(entry["fact1"]),
                        self.normalize_fact(entry["fact2"]),
                    ]
                    if split == "train"
                    else [],
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

    def parse_reference(self, path):
        reference = {}
        with open(path, "r") as file:
            for i, line in zip(range(300), file):
                id_, label = line.strip("\n").split(",")
                id_ = id_.strip('"')
                label = ["A", "B", "C", "D", "E", "F", "G", "H"].index(label.strip('"'))
                reference[id_] = label
        return reference

    def generate_relevant_question_ids(self):
        return {
            "train": self.find_relevant_questions(self.train_data),
            "validate": self.find_relevant_questions(self.validate_data),
            "test": self.find_relevant_questions(self.test_data),
        }

    def find_relevant_questions(self, data, find_range=3):
        embedder = Embedder()
        embeddings = embedder.embed([d["text_question"] for d in data])
        similarity = t.mm(embeddings, embeddings.transpose(0, 1))
        # exclude self
        similarity.fill_diagonal_(0)
        relevant_data = {}
        for idx, d in enumerate(data):
            rel_data = []
            for compare_idx in range(idx - find_range, idx + find_range + 1):
                if 0 <= compare_idx < len(data) and compare_idx != idx:
                    rel_data.append((similarity[idx, compare_idx], data[compare_idx]))
            rel_data = sorted(rel_data, key=lambda x: x[0], reverse=True)
            relevant_data[d["id"]] = [x[1] for x in rel_data]
        return relevant_data

    def validate_qasc_corpus_retrieval_rate(self):
        for split, data in zip(
            ("train", "validate"), (self.train_data, self.validate_data)
        ):
            fact1_retrieved_count = 0
            fact2_retrieved_count = 0
            fact1_allowed_count = 0
            fact2_allowed_count = 0
            for d in data:
                if d["original_facts"][0] in self.matcher.added_qasc_corpus_facts:
                    fact1_retrieved_count += 1
                if d["original_facts"][1] in self.matcher.added_qasc_corpus_facts:
                    fact2_retrieved_count += 1
                if (
                    self.matcher.fact_to_composite_node.get(d["original_facts"][0], -1)
                    in self.matcher.allowed_composite_nodes[d["id"]]
                ):
                    fact1_allowed_count += 1
                if (
                    self.matcher.fact_to_composite_node.get(d["original_facts"][1], -1)
                    in self.matcher.allowed_composite_nodes[d["id"]]
                ):
                    fact2_allowed_count += 1
            logging.info(
                f"All facts: {split}-Fact1 retrieved ratio: {fact1_retrieved_count / len(data)}"
            )
            logging.info(
                f"All facts: {split}-Fact2 retrieved ratio: {fact2_retrieved_count / len(data)}"
            )
            logging.info(
                f"All facts: {split}-retrieved length: {(fact1_retrieved_count + fact2_retrieved_count) / len(data)}"
            )
            logging.info(
                f"Allowed facts: {split}-Fact1 retrieved ratio: {fact1_allowed_count / len(data)}"
            )
            logging.info(
                f"Allowed facts: {split}-Fact2 retrieved ratio: {fact2_allowed_count / len(data)}"
            )
            logging.info(
                f"Allowed facts: {split}-retrieved length: {(fact1_allowed_count + fact2_allowed_count) / len(data)}"
            )

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


class QASCAugmentDataset(QASCBaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        use_augment: bool,
        augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
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

        super(QASCAugmentDataset, self).__init__(tokenizer)

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
            t5_input = normalize_t5_input(
                (
                    ", ".join(self.get_augment_context(split, data["id"]))
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
                normalize_t5_input(
                    ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"][
                        data["label"]
                    ]
                ),
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
            data["t5_label"] = normalize_t5_input(
                ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)"][data["label"]]
            )
        else:
            if self.use_augment:
                segments = [
                    [", ".join(self.get_augment_context(split, data["id"]))]
                    * len(data["choices"]),
                    [
                        " Q: "
                        + self.normalize_question(data["text_question"])
                        + " A: "
                        + ch
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
