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
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.download import SocialIQA
from encoder.dataset.matcher.social_iqa import SocialIQAMatcher
from encoder.dataset.matcher.fact_selector import FactSelector
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.file import PickleCache, open_file_with_create_directories
from encoder.models.embedder import Embedder
from .base import StaticIterableDataset
from .utils import normalize_t5_input


class SocialIQABaseDataset:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizerBase, None],
        match_closest_when_no_equal: bool = True,
    ):
        self.tokenizer = tokenizer
        self.match_closest_when_no_equal = match_closest_when_no_equal

        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.matcher = SocialIQAMatcher(tokenizer=self.matcher_tokenizer)
        self.social_iqa = SocialIQA().require()

        with PickleCache(
            os.path.join(preprocess_cache_dir, "social_iqa.data"), self.generate_data,
        ) as cache:
            self.train_data = [
                d for d in cache.data["train"] if len(d["text_question"]) < 300
            ]
            print(
                f"Removed {len(cache.data['train']) - len(self.train_data)} "
                f"train samples with very long contexts"
            )
            self.validate_data = cache.data["validate"]
            self.test_data = cache.data["test"]
            labels = [
                3,
                3,
                2,
                3,
                3,
                1,
                1,
                2,
                2,
                2,
                1,
                3,
                3,
                2,
                1,
                3,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                2,
                2,
                2,
                3,
                2,
                3,
            ]
            for i in range(30):
                self.test_data[i]["label"] = labels[i] - 1

        with PickleCache(
            os.path.join(preprocess_cache_dir, "social_iqa_selected_knowledge.data"),
            self.select_knowledge,
        ) as cache:
            question_allowed_knowledge, select_knowledge, mask = cache.data
            self.matcher.add_atomic_knowledge(
                question_allowed_knowledge, select_knowledge, mask
            )
        with PickleCache(
            os.path.join(preprocess_cache_dir, "social_iqa_selected_facts.data"),
            self.select_facts,
        ) as cache:
            question_facts = cache.data
            # Select the most similar one as the intermediate fact
            for data in chain(self.train_data, self.validate_data):
                data["facts"] = question_facts[data["id"]]

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

    @property
    def test_ref_dataset(self):
        return StaticIterableDataset(30, self.generator, ("test",))

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
            os.path.join(directory, "social_iqa.lst"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["1\n", "2\n", "3\n"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(answer_keys[label])

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "social_iqa.lst"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["1\n", "2\n", "3\n"]
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
            "train": self.parse_data(
                "train", self.social_iqa.train_path, self.social_iqa.train_labels_path
            ),
            "validate": self.parse_data(
                "validate",
                self.social_iqa.validate_path,
                self.social_iqa.validate_labels_path,
            ),
            "test": self.parse_data("test", self.social_iqa.test_path),
        }

    def parse_data(self, split, dataset_path, labels_path=None):
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
            text_choices = self.generate_choice_str(
                [entry["answerA"], entry["answerB"], entry["answerC"]]
            )

            choices = [entry["answerA"], entry["answerB"], entry["answerC"]]
            preprocessed = {
                "text_question": entry["context"] + " " + entry["question"],
                "text_choices": text_choices,
                "choices": choices,
                "choice_masks": self.generate_choice_match_mask(choices),
                "context": entry["context"],
                "question": entry["question"],
                "id": split + "|" + str(idx),
                "facts": [],
            }
            if labels_path is not None:
                # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                label = labels[idx]
                preprocessed["label"] = label
                preprocessed["text_answer"] = choices[label]

            data.append(preprocessed)
        return data

    def generate_choice_match_mask(self, choices: List[str]):
        words = []

        for choice in choices:
            words.append(
                set(
                    w.lower()
                    for w, pos in nltk.pos_tag(nltk.word_tokenize(choice))
                    if len(w) > 0
                    and (
                        pos.startswith("NN")
                        or pos.startswith("JJ")
                        or (
                            pos.startswith("VB")
                            and w.lower() not in self.matcher.VERB_FILTER_SET
                        )
                    )
                )
            )
        different_words = [
            words[0].difference(words[1].union(words[2])),
            words[1].difference(words[0].union(words[2])),
            words[2].difference(words[0].union(words[1])),
        ]
        masks = [["-"] * len(choice) for choice in choices]
        for diff_words, mask, choice in zip(different_words, masks, choices):
            if len(diff_words) == 0:
                mask[:] = ["+" for _ in range(len(mask))]
            choice = choice.lower()
            for word in diff_words:
                start = 0
                while True:
                    start = choice.find(word, start)
                    if start != -1:
                        mask[start : start + len(word)] = ["+"] * len(word)
                        start += len(word)
                    else:
                        break
        return ["".join(mask) for mask in masks]

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

    def select_facts(self):
        contexts = []
        choices = []
        query_ids = []
        for data in chain(self.train_data, self.validate_data):
            contexts.append(data["context"])
            choices.append(data["text_answer"])
            query_ids.append(data["id"])
        texts, _ = self.matcher.get_atomic_knowledge_text_and_mask()

        selected_facts = self.find_relative_knowledge_by_choice(
            contexts, choices, texts
        )

        return {
            id_: [allowed_facts[0]] if allowed_facts else []
            for id_, allowed_facts in zip(query_ids, selected_facts)
        }

    def select_knowledge(self):
        contexts = []
        choices = []
        query_ids = []
        for data in chain(self.train_data, self.validate_data, self.test_data):
            for choice in data["choices"]:
                contexts.append(data["context"])
                choices.append(choice)
                query_ids.append(data["id"])
        texts, masks = self.matcher.get_atomic_knowledge_text_and_mask()
        text_to_index = {t: idx for idx, t in enumerate(texts)}

        selected_facts = self.find_relative_knowledge_by_choice(
            contexts, choices, texts
        )

        all_selected_facts = sorted(
            list(set([f for facts in selected_facts for f in facts]))
        )
        all_selected_facts_mask = [masks[text_to_index[f]] for f in all_selected_facts]
        allowed_facts = {}
        for id_, query_allowed_facts in zip(query_ids, selected_facts):
            if id_ not in allowed_facts:
                allowed_facts[id_] = query_allowed_facts
            else:
                allowed_facts[id_] += query_allowed_facts
        return (
            allowed_facts,
            all_selected_facts,
            all_selected_facts_mask,
        )

    def find_relative_knowledge_by_choice(self, contexts, choices, texts):
        embedder = Embedder("cuda:0")
        fact_selector = FactSelector(
            [ctx + " " + ch for ctx, ch in zip(contexts, choices)],
            texts,
            min_score=0.4,
            max_facts=100,
        )
        all_selected_facts = []
        print("Computing embeddings for contexts")
        all_context_embeddings = embedder.embed(choices, show_prog_bar=True)
        print("Computing embeddings for first level facts")
        raw_selected_facts = sorted(
            list(set([f for sf in fact_selector.selected_facts for f in sf]))
        )
        raw_selected_facts_idx = {f: idx for idx, f in enumerate(raw_selected_facts)}
        raw_selected_facts_embeddings = embedder.embed(
            raw_selected_facts, show_prog_bar=True
        )
        for idx, (context, selected_facts) in tqdm(
            enumerate(zip(contexts, fact_selector.selected_facts)), total=len(contexts)
        ):
            if len(selected_facts) == 0:
                all_selected_facts.append([])
            else:
                context_embedding = all_context_embeddings[idx : idx + 1]
                fact_embeddings = t.cat(
                    [
                        raw_selected_facts_embeddings[
                            raw_selected_facts_idx[f] : raw_selected_facts_idx[f] + 1
                        ]
                        for f in selected_facts
                    ],
                    dim=0,
                )
                similarity = t.sum(context_embedding * fact_embeddings, dim=1)
                all_selected_facts.append(
                    [
                        selected_facts[i]
                        for i in t.topk(
                            similarity, k=min(10, len(selected_facts))
                        ).indices
                    ]
                )
        return all_selected_facts


class SocialIQAAugmentDataset(SocialIQABaseDataset):
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

        super(SocialIQAAugmentDataset, self).__init__(tokenizer)
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
                [data["text_question"] + " " + ch for ch in data["choices"]],
                # [data["text_question"]] * len(data["choices"]),
                # data["choices"],
                truncation="only_first",
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            data["sentence"] = encoded_sentence.input_ids.unsqueeze(0)
            data["mask"] = encoded_sentence.attention_mask.unsqueeze(0)
            data["type_ids"] = encoded_sentence.token_type_ids.unsqueeze(0)
        return data

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
