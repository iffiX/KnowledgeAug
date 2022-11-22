import itertools
import os
import copy
import json
import nltk
import random
import difflib
import logging
import numpy as np
import torch as t
from tqdm import tqdm
from itertools import chain
from typing import List, Tuple, Dict, Union
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.download import ANLI
from encoder.dataset.matcher.anli import ANLIMatcher
from encoder.dataset.matcher.fact_selector import FactSelector
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.file import JSONCache, PickleCache, open_file_with_create_directories
from .base import StaticIterableDataset
from .utils import normalize_t5_input

import multiprocessing as mp


class ANLIPathCreator:
    instance = None  # type: ANLIPathCreator

    def __init__(
        self,
        data: List[Tuple[str, str, str, str]],
        matcher: ANLIMatcher,
        max_depth: int = 2,
    ):
        self.data = data
        self.matcher = matcher
        self.max_depth = max_depth

    @staticmethod
    def get_path_for_sample(idx):
        self = ANLIPathCreator.instance
        result1 = self.matcher.find_shortest_path(
            source_sentence=self.data[idx][1],
            target_sentence=self.data[idx][2],
            find_target=True,
            max_depth_for_each_node=self.max_depth,
            min_levels_before_checking_target_reached=0,
        )
        result2 = self.matcher.find_shortest_path(
            source_sentence=self.data[idx][2],
            target_sentence=self.data[idx][3],
            find_target=True,
            max_depth_for_each_node=self.max_depth,
            min_levels_before_checking_target_reached=0,
        )
        l1_path = ", ".join(", ".join(res) for res in result1[0])
        l2_path = ", ".join(", ".join(res) for res in result2[0])
        return self.data[idx][0], " # ".join([l1_path, l2_path])

    @staticmethod
    def initialize_pool(data, matcher, max_depth):
        ANLIPathCreator.instance = ANLIPathCreator(data, matcher, max_depth)


class ANLIPathGenerator:
    def __init__(
        self,
        data: List[Tuple[str, str, str, str]],
        matcher: ANLIMatcher,
        max_depth: int = 2,
    ):
        """
        Args:
            data: id, hypothesis1, choice, hypothesis2
        """
        self.data, self.matcher, self.max_depth = (
            data,
            matcher,
            max_depth,
        )
        self.paths = self.generate_paths()

    def generate_paths(self):
        result = {}
        with mp.Pool(
            initializer=ANLIPathCreator.initialize_pool,
            initargs=(self.data, self.matcher, self.max_depth),
        ) as pool, tqdm(total=len(self.data)) as pbar:
            for (_id, transitions) in pool.imap_unordered(
                ANLIPathCreator.get_path_for_sample, range(len(self.data)),
            ):
                pbar.update()
                result[_id] = transitions
        return result


class ANLIBaseDataset:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        match_closest_when_no_equal: bool = True,
    ):
        self.tokenizer = tokenizer
        self.match_closest_when_no_equal = match_closest_when_no_equal

        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.matcher = ANLIMatcher(tokenizer=self.matcher_tokenizer)
        self.anli = ANLI().require()

        with PickleCache(
            os.path.join(preprocess_cache_dir, "anli.data"), self.generate_data,
        ) as cache:
            random.Random(42).shuffle(cache.data["train"])
            self.train_data = [
                d for d in cache.data["train"] if d["choices"][0] != d["choices"][1]
            ]
            self.validate_data = cache.data["validate"]
            self.test_data = cache.data["test"]

        self.generate_and_add_paths()
        # self.generate_and_add_knowledge()
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
            answer = batch["choices"][i][labels[i]]
            ref_answer = batch["choices"][i][batch["label"][i]]

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
            os.path.join(directory, "anli.lst"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["1\n", "2\n"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(answer_keys[label])

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "anli.lst"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["1\n", "2\n"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(preprocessed["choices"]):
                    if answer == choice:
                        file.write(answer_keys[i])
                        break
                else:
                    missing += 1
                    print(
                        f"Missing answer, choices: {preprocessed['choices']}, "
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
            "train": self.parse_data(self.anli.train_path, self.anli.train_labels_path),
            "validate": self.parse_data(
                self.anli.validate_path, self.anli.validate_labels_path
            ),
            "test": self.parse_data(self.anli.test_path),
        }

    # def generate_and_add_knowledge(self):
    #     def generate():
    #         queries = []
    #         query_ids = []
    #         for data in chain(self.train_data, self.validate_data, self.test_data,):
    #             id_ = data["id"]
    #             queries.append(data["choices"][0])
    #             queries.append(data["choices"][1])
    #             query_ids += [id_, id_]
    #
    #         texts, masks = self.matcher.get_atomic_knowledge_text_and_mask()
    #         text_to_index = {t: idx for idx, t in enumerate(texts)}
    #
    #         fact_selector = FactSelector(
    #             queries, texts, max_facts=3, inner_batch_size=1024
    #         )
    #
    #         all_selected_facts = sorted(
    #             list(set([f for facts in fact_selector.selected_facts for f in facts]))
    #         )
    #         all_selected_facts_mask = [
    #             masks[text_to_index[f]] for f in all_selected_facts
    #         ]
    #         return (
    #             fact_selector.selected_facts,
    #             all_selected_facts,
    #             all_selected_facts_mask,
    #         )
    #
    #     with PickleCache(
    #         os.path.join(
    #             preprocess_cache_dir, "anli_augment_generator_selected_facts.data"
    #         ),
    #         generate,
    #     ) as cache:
    #         selected_facts, all_selected_facts, all_selected_facts_mask = cache.data
    #
    #     self.matcher.add_atomic_knowledge(all_selected_facts, all_selected_facts_mask)
    #     for idx, data in enumerate(
    #         chain(self.train_data, self.validate_data, self.test_data,)
    #     ):
    #         potential_facts = selected_facts[idx * 2 + data["label"]]
    #         data["facts"] = [potential_facts[0]] if potential_facts else []

    def parse_data(self, dataset_path, labels_path=None):
        data = []
        labels = []
        logging.info(f"Parsing {dataset_path}")
        if labels_path is not None:
            with open(labels_path) as file:
                for line in file:
                    labels.append(int(line.strip("\n")) - 1)
        all_entries = []
        with open(dataset_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                all_entries.append((idx, entry))

        for idx, entry in tqdm(all_entries):
            text_choices = self.generate_choice_str([entry["hyp1"], entry["hyp2"]])

            choices = [entry["hyp1"], entry["hyp2"]]
            diff, mask = self.get_different_parts(entry["hyp1"], entry["hyp2"])
            preprocessed = {
                "text_question": entry["obs1"] + " " + entry["obs2"],
                "text_choices": text_choices,
                "obs1": entry["obs1"],
                "obs2": entry["obs2"],
                "choices": choices,
                "choice_masks": mask,
                "diff_choices": diff,
                "id": entry["story_id"] + "|" + str(idx),
                "story_id": entry["story_id"],
                "facts": [],
                "paths": [],
            }
            if labels_path is not None:
                # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                label = labels[idx]
                preprocessed["label"] = label
                preprocessed["text_answer"] = choices[label]

            data.append(preprocessed)
        return data

    def get_different_parts(self, choice1, choice2):
        lemmatizer = WordNetLemmatizer()
        lemma_map = {"N": wordnet.NOUN, "J": wordnet.ADJ, "V": wordnet.VERB}
        raw_words1 = [
            (w, pos)
            for w, pos in nltk.pos_tag(nltk.word_tokenize(choice1))
            if pos.startswith("NN")
            or pos.startswith("JJ")
            or (pos.startswith("VB") and w.lower() not in self.matcher.VERB_FILTER_SET)
        ]
        raw_words2 = [
            (w, pos)
            for w, pos in nltk.pos_tag(nltk.word_tokenize(choice2))
            if pos.startswith("NN")
            or pos.startswith("JJ")
            or (pos.startswith("VB") and w.lower() not in self.matcher.VERB_FILTER_SET)
        ]
        words1_map = {
            lemmatizer.lemmatize(w, lemma_map[pos[0]]).lower(): w
            for w, pos in raw_words1
        }
        words2_map = {
            lemmatizer.lemmatize(w, lemma_map[pos[0]]).lower(): w
            for w, pos in raw_words2
        }
        words1 = set(words1_map.keys())

        words2 = set(words2_map.keys())

        selected_words1 = words1.difference(words2)
        selected_words1 = sorted(list(w for w in selected_words1 if len(w) > 0))
        selected_words2 = words2.difference(words1)
        selected_words2 = sorted(list(w for w in selected_words2 if len(w) > 0))
        mask1 = ["-"] * len(choice1)
        mask2 = ["-"] * len(choice2)
        choice1 = choice1.lower()
        choice2 = choice2.lower()
        for word in selected_words1:
            start = 0
            word = words1_map[word]
            while True:
                start = choice1.find(word, start)
                if start != -1:
                    mask1[start : start + len(word)] = ["+"] * len(word)
                    start += len(word)
                else:
                    break
        for word in selected_words2:
            start = 0
            word = words2_map[word]
            while True:
                start = choice2.find(word, start)
                if start != -1:
                    mask2[start : start + len(word)] = ["+"] * len(word)
                    start += len(word)
                else:
                    break
        return (
            (" ".join(selected_words1), " ".join(selected_words2),),
            ("".join(mask1), "".join(mask2)),
        )

    def generate_and_add_paths(self):
        all_data = [
            d
            for d in itertools.chain(
                self.train_data, self.validate_data, self.test_data
            )
            if len(d["diff_choices"][0]) > 0 and len(d["diff_choices"][1]) > 0
        ]

        def generate():
            choice1_paths = ANLIPathGenerator(
                [
                    (d["id"], d["obs1"], d["diff_choices"][0], d["obs2"])
                    for d in all_data
                ],
                self.matcher,
                2,
            ).paths
            choice2_paths = ANLIPathGenerator(
                [
                    (d["id"], d["obs1"], d["diff_choices"][1], d["obs2"])
                    for d in all_data
                ],
                self.matcher,
                2,
            ).paths
            paths = {}
            for key in choice1_paths.keys():
                paths[key] = [choice1_paths[key], choice2_paths[key]]
            return paths

        with JSONCache(
            os.path.join(preprocess_cache_dir, "anli_paths.json"), generate
        ) as cache:
            for d in all_data:
                d["paths"] = cache.data[d["id"]]

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


class ANLIAugmentDataset(ANLIBaseDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        # augment_contexts: Tuple[Dict[str, List[str]], Dict[str, List[str]]],
        max_seq_length: int = 300,
        output_mode: str = "single",
    ):
        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")

        # self.augment_contexts = augment_contexts
        self.max_seq_length = max_seq_length
        self.output_mode = output_mode
        self.rand = random.Random(42)
        self.rand_train = True

        super(ANLIAugmentDataset, self).__init__(tokenizer)
        # for split, split_data in (
        #     ("train", self.train_data),
        #     ("validate", self.validate_data),
        #     ("test", self.test_data),
        # ):
        #     found_count = sum(
        #         1 for data in split_data if data["id"] in augment_contexts[0]
        #     )
        #     print(
        #         f"{found_count}/{len(split_data)} samples of {split} split have contexts"
        #     )

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
                # [", ".join(self.get_augment_context(split, data["id"]))]
                # * len(data["choices"]),
                [
                    "Path for choice 1: "
                    + data["paths"][0]
                    + " Path for choice 2: "
                    + data["paths"][1]
                ]
                * 2
                if data["paths"]
                else ["", ""],
                [
                    "Question: "
                    + self.normalize_question(data["text_question"])
                    + " Choices: "
                    + ", ".join(data["choices"])
                    + " Answer: "
                    + ch
                    for ch in data["choices"]
                ],
                truncation="only_first",
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            # encoded_sentence = self.tokenizer(
            #     [self.normalize_question(data["text_question"])] * len(data["choices"]),
            #     [
            #         " Choices: " + ", ".join(data["choices"]) + " Answer: " + ch
            #         for ch in data["choices"]
            #     ],
            #     truncation="only_first",
            #     padding="max_length",
            #     max_length=self.max_seq_length,
            #     return_tensors="pt",
            # )
            data["sentence"] = encoded_sentence.input_ids.unsqueeze(0)
            data["mask"] = encoded_sentence.attention_mask.unsqueeze(0)
            data["type_ids"] = encoded_sentence.token_type_ids.unsqueeze(0)
        return data

    def normalize_question(self, question):
        if question.endswith("?") or question.endswith("."):
            return question.capitalize()
        else:
            return (question + ".").capitalize()

    def get_augment_context(self, split, data_id, no_rand=False):
        # if split != "train":
        #     augment_context = self.augment_contexts[0].get(data_id, [])
        # else:
        #     if self.rand_train and not no_rand:
        #         ref_context = self.augment_contexts[1].get(data_id)
        #         augment_context = self.augment_contexts[0].get(data_id, None)
        #         if augment_context is not None:
        #             augment_context = self.rand.choice([ref_context, augment_context])
        #         else:
        #             augment_context = ref_context
        #     else:
        #         augment_context = self.augment_contexts[1].get(data_id)
        augment_context = self.augment_contexts[0].get(data_id, [])
        return augment_context
