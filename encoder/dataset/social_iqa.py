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
            for i in range(len(self.test_data)):
                if i < 30:
                    self.test_data[i]["label"] = labels[i] - 1
                else:
                    self.test_data[i]["label"] = -1

        with PickleCache(
            os.path.join(preprocess_cache_dir, "social_iqa_selected_knowledge.data"),
            self.select_knowledge,
        ) as cache:
            question_allowed_knowledge, select_knowledge, mask = cache.data
            self.matcher.add_atomic_knowledge(
                question_allowed_knowledge, select_knowledge, mask
            )
        self.question_allowed_knowledge = question_allowed_knowledge

        with PickleCache(
            os.path.join(preprocess_cache_dir, "social_iqa_selected_facts.data"),
            self.select_facts,
        ) as cache:
            question_facts = cache.data
            # Select the most similar one as the intermediate fact
            for data in chain(self.train_data, self.validate_data):
                if not question_allowed_knowledge[data["id"]]:
                    question_facts[data["id"]] = []
            for data in chain(self.train_data, self.validate_data):
                data["facts"] = question_facts[data["id"]]

        self.validate_answer_retreival(
            self.train_data, question_facts, question_allowed_knowledge
        )
        self.validate_answer_retreival(
            self.validate_data, question_facts, question_allowed_knowledge
        )
        self.validate_answer_retreival(self.test_data, None, question_allowed_knowledge)
        self.set_corpus()

    def validate_answer_retreival(self, data, facts, allowed_knowledge):
        fact_retrieval_count = 0
        allowed_retrieval_count = 0
        has_fact_count = 0
        has_allowed_count = 0
        for d in data:
            words = d["choice_diffs"][d["label"]]
            if facts is not None:
                if facts[d["id"]]:
                    fact = facts[d["id"]][0]
                    has_fact_count += 1
                    if any(word in fact for word in words):
                        fact_retrieval_count += 1
            if allowed_knowledge[d["id"]]:
                allowed = allowed_knowledge[d["id"]]
                has_allowed_count += 1
                for f in allowed:
                    if any(word in f for word in words):
                        allowed_retrieval_count += 1
                        break

        if facts is not None:
            print(
                f"correct choice in fact rate: {fact_retrieval_count / has_fact_count}"
            )
        print(
            f"correct choice in allowed knowledge rate: {allowed_retrieval_count / has_allowed_count}"
        )
        if facts is not None:
            print(f"total samples with fact count: {has_fact_count}")
        print(f"total samples with allowed knowledge count: {has_allowed_count}")

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
        choices = [normalize_t5_input(x) for x in ["(A)", "(B)", "(C)"]]
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
        choices = [normalize_t5_input(x) for x in ["(A)", "(B)", "(C)"]]
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
                for i, choice in enumerate(choices):
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
            masks, diff_parts = self.generate_choice_match_mask(choices)
            preprocessed = {
                "text_question": entry["context"] + " " + entry["question"],
                "text_choices": text_choices,
                "choices": choices,
                "choice_masks": masks,
                "choice_diffs": diff_parts,
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
        return ["".join(mask) for mask in masks], different_words

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
        questions = []
        choices = []
        query_ids = []
        for data in chain(self.train_data, self.validate_data):
            contexts.append(data["context"])
            questions.append(data["question"])
            choices.append(data["text_answer"])
            query_ids.append(data["id"])

        selected_facts = self.find_related_knowledge(
            contexts,
            questions,
            choices,
            context_similarity=0.35,
            choice_similarity=0.55,
        )

        return {
            id_: [allowed_facts[0]] if allowed_facts else []
            for id_, allowed_facts in zip(query_ids, selected_facts)
        }

    def select_knowledge(self):
        contexts = []
        questions = []
        choices = []
        query_ids = []
        for data in chain(self.train_data, self.validate_data, self.test_data):
            for choice in data["choices"]:
                contexts.append(data["context"])
                questions.append(data["question"])
                choices.append(choice)
                query_ids.append(data["id"])
        texts, masks, _ = self.matcher.get_atomic_knowledge_text_and_mask()
        text_to_index = {t: idx for idx, t in enumerate(texts)}

        selected_facts = self.find_related_knowledge(
            contexts,
            questions,
            choices,
            context_similarity=0.35,
            choice_similarity=0.55,
        )

        all_selected_facts = sorted(
            list(set([f for facts in selected_facts for f in facts]))
        )
        all_selected_facts_mask = [masks[text_to_index[f]] for f in all_selected_facts]
        allowed_facts = {}
        choice_with_allowed_facts_count = {}
        for id_, query_allowed_facts in zip(query_ids, selected_facts):
            if id_ not in allowed_facts:
                allowed_facts[id_] = query_allowed_facts[:2]
                choice_with_allowed_facts_count[id_] = 1 if query_allowed_facts else 0
            else:
                allowed_facts[id_] += query_allowed_facts[:2]
                choice_with_allowed_facts_count[id_] += 1 if query_allowed_facts else 0
        for id_, count in choice_with_allowed_facts_count.items():
            if count < 2:
                allowed_facts[id_] = []
        return (
            allowed_facts,
            all_selected_facts,
            all_selected_facts_mask,
        )

    def find_related_knowledge(
        self, contexts, questions, choices, context_similarity, choice_similarity,
    ):
        embedder = Embedder("cuda:0")
        (
            knowledge,
            mask,
            relation_to_triple,
        ) = self.matcher.get_atomic_knowledge_text_and_mask()

        def generate_atomic_knowledge_embedding():
            triple_embedding = {}
            print("Computing embeddings for atomic knowledge")
            for relation, triples in relation_to_triple.items():
                print(f"Processing relation {relation}")
                triple_embedding[relation] = (
                    embedder.embed([t[0] for t in triples], show_prog_bar=True),
                    embedder.embed([t[1] for t in triples], show_prog_bar=True),
                    [t[2] for t in triples],
                )
            return triple_embedding

        with PickleCache(
            os.path.join(preprocess_cache_dir, "social_iqa_atomic_embedding.data"),
            generate_atomic_knowledge_embedding,
        ) as cache:
            triple_embedding = cache.data

        print("Computing embeddings for contexts")
        context_embeddings = embedder.embed(contexts, show_prog_bar=True)
        print("Computing embeddings for choices")
        choice_embeddings = embedder.embed(choices, show_prog_bar=True)

        selected_facts = []
        for context_embedding, question, choice_embedding in tqdm(
            zip(context_embeddings, questions, choice_embeddings), total=len(contexts)
        ):
            question = question.lower()
            if re.search("others feel", question):
                question_type = "oReact"
            elif re.search("others want", question):
                question_type = "oWant"
            elif re.search("happen to others", question):
                question_type = "oEffect"
            elif re.search("feel", question):
                question_type = "xReact"
            elif re.search("want to do", question):
                question_type = "xWant"
            elif re.search("happen to", question):
                question_type = "xEffect"
            elif re.search("describe", question):
                question_type = "xAttr"
            elif re.search("need to do", question):
                question_type = "xNeed"
            elif re.search("why did", question):
                question_type = "xIntent"
            else:
                selected_facts.append([])
                continue

            context_similarities = t.sum(
                triple_embedding[question_type][0] * context_embedding, dim=1
            )
            choice_similarities = t.sum(
                triple_embedding[question_type][1] * choice_embedding, dim=1
            )
            similarities = context_similarities + choice_similarities
            selected_facts.append(
                [
                    triple_embedding[question_type][2][i]
                    for i in t.topk(
                        similarities, k=min(10, similarities.shape[0])
                    ).indices
                    if context_similarities[i] > context_similarity
                    and choice_similarities[i] > choice_similarity
                    and similarities[i] > context_similarity + choice_similarity
                ]
            )
        return selected_facts


class SocialIQAAugmentDataset(SocialIQABaseDataset):
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

        super(SocialIQAAugmentDataset, self).__init__(tokenizer)
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
                + data["text_question"].replace("...\n", "").replace("\n", "")
                + " \\n "
                + data["text_choices"].replace("...\n", "").replace("\n", "")
            )
            encoded_sentence = self.tokenizer(
                t5_input,
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )

            answer = self.tokenizer.encode(
                normalize_t5_input(["(A)", "(B)", "(C)"][data["label"]]),
                padding="max_length",
                max_length=32,
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
            data["t5_label"] = normalize_t5_input(["(A)", "(B)", "(C)"][data["label"]])
        else:
            if self.use_augment:
                try:
                    if "\n" in data["text_question"]:
                        data["text_question"] = (
                            data["text_question"].replace("...\n", "").replace("\n", "")
                        )
                        data["choices"] = [
                            ch.replace("...\n", "").replace("\n", "")[:100]
                            for ch in data["choices"]
                        ]
                    seq1 = [", ".join(self.augment_contexts.get(data["id"], []))] * len(
                        data["choices"]
                    )
                    seq2 = [
                        "Question: "
                        + data["text_question"]
                        + " Choices: "
                        + ", ".join(data["choices"])
                        + " Answer: "
                        + ch
                        for ch in data["choices"]
                    ]
                    segments = [seq1, seq2]
                    encoded_sentence = self.tokenizer(
                        *segments,
                        truncation="only_first",
                        padding="max_length",
                        max_length=self.max_seq_length,
                        return_tensors="pt",
                    )
                except Exception as e:
                    raise e
            else:
                segments = [
                    [data["text_question"]] * len(data["choices"]),
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
