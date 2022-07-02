import os
import copy
import json
import pickle
import logging
import random
import torch as t
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.utils.settings import (
    preprocess_cache_dir,
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.file import open_file_with_create_directories
from .base import StaticIterableDataset
from .download import OpenBookQA, QASC, ARC


class NegativeFactSampler:
    def __init__(self, fact_num, target_fact_idx, seed=42):
        self.target_fact_idx = target_fact_idx
        self.generator = random.Random(seed)
        self.available_fact_idx = list(
            set(range(fact_num)).difference({self.target_fact_idx})
        )

    def __call__(self, exclude_indexes=None):
        available_fact_idx = self.available_fact_idx
        if exclude_indexes is not None:
            available_fact_idx = list(
                set(available_fact_idx).difference(set(exclude_indexes))
            )
        return self.generator.choice(available_fact_idx)


class ARCSearchDataset:
    def __init__(
        self,
        retriever_tokenizer: PreTrainedTokenizerBase,
        reranker_tokenizer: PreTrainedTokenizerBase,
        retriever_negative_samples: int = 4,
        retriever_max_seq_length: int = 100,
        reranker_negative_samples: int = 4,
        reranker_max_seq_length: int = 100,
        seed: int = 42,
    ):
        self.retriever_tokenizer = retriever_tokenizer
        self.reranker_tokenizer = reranker_tokenizer
        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.retriever_negative_samples = retriever_negative_samples
        self.retriever_max_seq_length = retriever_max_seq_length
        self.reranker_negative_samples = reranker_negative_samples
        self.reranker_max_seq_length = reranker_max_seq_length
        self.openbook_qa = OpenBookQA().require()
        self.arc = ARC().require()
        self.qasc = QASC().require()

        archive_path = os.path.join(preprocess_cache_dir, "arc_search.data")
        if not os.path.exists(archive_path):
            self.facts = self.generate_facts()
            (
                self.train_retriever_data,
                self.unannotated_train_data,
                self.search_data,
            ) = self.generate_retriever_train_data()
            self.train_reranker_data = self.generate_reranker_train_data()
            self.validate_data = self.generate_validate_data()
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
                self.facts = data["facts"]
                self.train_retriever_data = data["train_retriever"]
                self.train_reranker_data = data["train_reranker"]
                self.unannotated_train_data = data["unannotated_train"]
                self.search_data = data["search"]
                self.validate_data = data["validate"]

        self.random = random.Random(seed)
        self.negative_fact_gen = {
            fact: NegativeFactSampler(
                len(self.facts), self.facts.index(fact), seed=seed + idx
            )
            for idx, fact in enumerate(self.facts)
        }

        (
            self.retriever_all_fact_sentences,
            self.retriever_all_fact_masks,
            self.retriever_all_fact_type_ids,
        ) = (
            [],
            [],
            [],
        )
        for fact in self.facts:
            encoded_sentence = self.retriever_tokenizer(
                fact,
                padding="max_length",
                max_length=self.retriever_max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            self.retriever_all_fact_sentences.append(encoded_sentence.input_ids)
            self.retriever_all_fact_masks.append(encoded_sentence.attention_mask)
            if hasattr(encoded_sentence, "token_type_ids"):
                self.retriever_all_fact_type_ids.append(encoded_sentence.token_type_ids)
        self.retriever_has_token_type_ids = len(self.retriever_all_fact_type_ids) > 0

    @property
    def facts_dataset(self):
        return StaticIterableDataset(
            len(self.retriever_all_fact_sentences), self.retriever_generator, ("facts",)
        )

    @property
    def train_retriever_dataset(self):
        return StaticIterableDataset(
            len(self.train_retriever_data), self.retriever_generator, ("train",),
        )

    @property
    def train_retriever_candidates_dataset(self):
        return StaticIterableDataset(
            len(self.train_retriever_data),
            self.retriever_generator,
            ("train_candidates",),
        )

    @property
    def train_reranker_dataset(self):
        return StaticIterableDataset(
            len(self.train_reranker_data), self.reranker_generator
        )

    @property
    def unannotated_train_dataset(self):
        return StaticIterableDataset(
            len(self.unannotated_train_data),
            self.retriever_generator,
            ("unannotated_train",),
        )

    @property
    def validate_dataset(self):
        return StaticIterableDataset(
            len(self.validate_data), self.retriever_generator, ("validate",)
        )

    @property
    def search_dataset(self):
        return StaticIterableDataset(
            len(self.search_data), self.retriever_generator, ("search",),
        )

    def retriever_generator(self, index: int, split: str):
        if split == "facts":
            data = {
                "sentence": self.retriever_all_fact_sentences[index].unsqueeze(1),
                "mask": self.retriever_all_fact_masks[index].unsqueeze(1),
                "id": f"facts_{index}",
            }
            if self.retriever_has_token_type_ids:
                data["type_ids"] = self.retriever_all_fact_type_ids[index].unsqueeze(1)
            return data
        if split == "train" or split == "train_candidates":
            data = self.train_retriever_data[index]
        elif split == "unannotated_train":
            data = self.unannotated_train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        elif split == "search":
            data = self.search_data[index]
        else:
            raise ValueError(f"Invalid split: {split}")

        data = copy.deepcopy(data)

        sentences, masks, type_ids = [], [], []
        correct_choice = -1
        if split == "train":
            correct_choice = self.random.randint(0, self.retriever_negative_samples)
            exclude_indexes = None
            if len(data["facts"]) > 1:
                exclude_indexes = [self.facts.index(f) for f in data["facts"][1:]]

            encoded_sentence = self.retriever_tokenizer(
                data["question"],
                ",".join(data["choices"]),
                padding="max_length",
                max_length=self.retriever_max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            sentences.append(encoded_sentence.input_ids)
            masks.append(encoded_sentence.attention_mask)
            if self.retriever_has_token_type_ids:
                type_ids.append(encoded_sentence.token_type_ids)
            for i in range(self.retriever_negative_samples + 1):
                fact_idx = (
                    self.facts.index(data["facts"][0])
                    if i == correct_choice
                    else self.negative_fact_gen[data["facts"][0]](exclude_indexes)
                )
                sentences.append(self.retriever_all_fact_sentences[fact_idx])
                masks.append(self.retriever_all_fact_masks[fact_idx])
                if self.retriever_has_token_type_ids:
                    type_ids.append(self.retriever_all_fact_type_ids[fact_idx])
        else:
            if split == "validate":
                correct_choice = self.facts.index(data["facts"][0])

            encoded_sentence = self.retriever_tokenizer(
                data["question"],
                ",".join(data["choices"]),
                padding="max_length",
                max_length=self.retriever_max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            sentences.append(encoded_sentence.input_ids)
            masks.append(encoded_sentence.attention_mask)
            if self.retriever_has_token_type_ids:
                type_ids.append(encoded_sentence.token_type_ids)

        data["sentence"] = t.stack(sentences, dim=1)
        data["mask"] = t.stack(masks, dim=1)
        if self.retriever_has_token_type_ids:
            data["type_ids"] = t.stack(type_ids, dim=1)
        if split in ("train", "validate"):
            data["answer"] = correct_choice
        return data

    def reranker_generator(self, index: int):
        data = copy.deepcopy(self.train_reranker_data[index])

        sentences, masks, type_ids = [], [], []
        if data["id"].startswith("arc"):
            # Use choices as potential "facts"
            correct_choice = data["choices"].index(data["facts"][0])
            for choice in data["choices"]:
                encoded_sentence = self.reranker_tokenizer(
                    data["question"],
                    choice,
                    padding="max_length",
                    max_length=self.reranker_max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)
        else:
            correct_choice = self.random.randint(0, self.reranker_negative_samples)
            use_candidates = self.random.random() > 0.5
            exclude_indexes = None
            if len(data["facts"]) > 1:
                exclude_indexes = [self.facts.index(f) for f in data["facts"][1:]]

            negative_count = 0
            for i in range(self.reranker_negative_samples + 1):
                if i == correct_choice:
                    fact_idx = self.facts.index(data["facts"][0])
                elif use_candidates and negative_count < len(
                    data["candidate_fact_indices"]
                ):
                    fact_idx = data["candidate_fact_indices"][negative_count]
                    negative_count += 1
                else:
                    fact_idx = self.negative_fact_gen[data["facts"][0]](exclude_indexes)
                    negative_count += 1

                encoded_sentence = self.reranker_tokenizer(
                    data["question"] + f" {','.join(data['choices'])}",
                    self.facts[fact_idx],
                    padding="max_length",
                    max_length=self.reranker_max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)

        data["sentence"] = t.stack(sentences, dim=1)
        data["mask"] = t.stack(masks, dim=1)
        data["type_ids"] = t.stack(type_ids, dim=1)
        data["answer"] = correct_choice
        return data

    def generate_reranker_input(self, top_k, split):
        if split == "unannotated_train":
            data_list = self.unannotated_train_data
        elif split == "validate":
            data_list = self.validate_data
        elif split == "search":
            data_list = self.search_data
        else:
            raise ValueError(f"Invalid split: {split}")
        if len(data_list) != len(top_k):
            raise ValueError(f"Data length and top k fact length does not match")
        data_list = copy.deepcopy(data_list)

        sentences, masks, type_ids = [], [], []
        for data, top_k_facts in zip(data_list, top_k):
            sub_sentences, sub_masks, sub_type_ids = [], [], []
            for fact_idx in top_k_facts:
                encoded_sentence = self.reranker_tokenizer(
                    data["question"] + f" {','.join(data['choices'])}",
                    self.facts[fact_idx],
                    padding="max_length",
                    max_length=self.reranker_max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                sub_sentences.append(encoded_sentence.input_ids)
                sub_masks.append(encoded_sentence.attention_mask)
                sub_type_ids.append(encoded_sentence.token_type_ids)
            sentences.append(t.stack(sub_sentences, dim=1))
            masks.append(t.stack(sub_masks, dim=1))
            type_ids.append(t.stack(sub_type_ids, dim=1))
        return t.cat(sentences), t.cat(masks), t.cat(type_ids)

    def validate_search(self, batch: BatchEncoding, choice: t.Tensor):
        total = choice.shape[0]
        total_correct = 0
        for i in range(choice.shape[0]):
            if choice.dim() == 1 or (choice.dim() == 2 and choice.shape[1] == 1):
                fact = self.facts[choice[i].item()]
                found = fact in batch["facts"][i]
                total_correct += found
                sentence = batch["question"][i] + f" ({batch['facts'][i][0]}) "
                if not found:
                    print(
                        f"sentence: [{sentence}] \n"
                        f"fact: [{fact}] \n"
                        f"ref_fact: [{'|'.join(batch['facts'][i])}] \n"
                    )
            else:
                found = False
                for sub_choice in choice[i]:
                    fact = self.facts[sub_choice.item()]
                    if fact in batch["facts"][i]:
                        total_correct += 1
                        found = True
                        break
                sentence = batch["question"][i] + f" ({batch['facts'][i][0]}) "
                if not found:
                    print(
                        f"sentence: [{sentence}] \n"
                        f"facts: [{'|'.join([self.facts[sub_choice.item()] for sub_choice in choice[i]])}] \n"
                        f"ref_fact: [{'|'.join(batch['facts'][i])}] \n"
                    )
        return {"accuracy": total_correct / total}

    def generate_retriever_train_data(self):
        data = []
        unannotated_data = []
        search_data = []
        logging.info("Generating retriever train data")
        with open(self.openbook_qa.train_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact = entry["fact1"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "choices": [c["text"] for c in entry["question"]["choices"]],
                    "facts": [fact],
                    "candidate_fact_indices": [],
                    "id": f"openbook_qa_train_{idx}",
                }
                data.append(preprocessed)
        with open(self.qasc.train_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact1 = entry["fact1"].strip(".").lower()
                fact2 = entry["fact2"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "choices": [c["text"] for c in entry["question"]["choices"]],
                    "facts": [fact1, fact2],
                    "candidate_fact_indices": [],
                    "id": f"qasc_train_{idx}",
                }
                data.append(preprocessed)
        for name, path in (
            ("arc_train_challenge", self.arc.train_challenge_path),
            ("arc_validate_challenge", self.arc.validate_challenge_path),
            ("arc_train_easy", self.arc.train_easy_path),
            ("arc_validate_easy", self.arc.validate_easy_path),
            ("arc_test_challenge", self.arc.test_challenge_path),
            ("arc_test_easy", self.arc.test_easy_path),
            ("qasc_test", self.qasc.test_path),
        ):
            with open(path, "r") as file:
                for idx, line in enumerate(file):
                    entry = json.loads(line)
                    preprocessed = {
                        "question": entry["question"]["stem"],
                        "choices": [c["text"] for c in entry["question"]["choices"]],
                        "facts": [],
                        "candidate_fact_indices": [],
                        "id": f"{name}_{idx}",
                    }
                    if "train" not in name and "validate" not in name:
                        unannotated_data.append(preprocessed)
                    if "arc" in name:
                        search_data.append(preprocessed)
        return data, unannotated_data, search_data

    def generate_reranker_train_data(self):
        data = []
        logging.info("Generating reranker train data")
        with open(self.openbook_qa.train_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact = entry["fact1"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "choices": [c["text"] for c in entry["question"]["choices"]],
                    "facts": [fact],
                    "candidate_fact_indices": [],
                    "id": f"openbook_qa_train_{idx}",
                }
                data.append(preprocessed)
        with open(self.qasc.train_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact1 = entry["fact1"].strip(".").lower()
                fact2 = entry["fact2"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "choices": [c["text"] for c in entry["question"]["choices"]],
                    "facts": [fact1, fact2],
                    "candidate_fact_indices": [],
                    "id": f"qasc_train_{idx}",
                }
                data.append(preprocessed)
        # for name, path in (
        #     ("arc_train_challenge", self.arc.train_challenge_path),
        #     ("arc_validate_challenge", self.arc.validate_challenge_path),
        #     ("arc_train_easy", self.arc.train_easy_path),
        #     ("arc_validate_easy", self.arc.validate_easy_path),
        # ):
        #     with open(path, "r") as file:
        #         for idx, line in enumerate(file):
        #             entry = json.loads(line)
        #             preprocessed = {
        #                 "question": entry["question"]["stem"],
        #                 "choices": [c["text"] for c in entry["question"]["choices"]],
        #                 "facts": [
        #                     c["text"]
        #                     for c in entry["question"]["choices"]
        #                     if c["label"] == entry["answerKey"]
        #                 ],
        #                 "candidate_fact_indices": [],
        #                 "id": f"{name}_{idx}",
        #             }
        #             data.append(preprocessed)
        return data

    def generate_validate_data(self):
        data = []
        logging.info("Generating validate data")
        for name, path in (
            ("openbook_qa_validate", self.openbook_qa.validate_path),
            ("openbook_qa_test", self.openbook_qa.test_path),
        ):
            with open(path, "r") as file:
                for idx, line in enumerate(file):
                    entry = json.loads(line)
                    fact = entry["fact1"].strip(".").lower()
                    preprocessed = {
                        "question": entry["question"]["stem"],
                        "choices": [c["text"] for c in entry["question"]["choices"]],
                        "facts": [fact],
                        "id": f"{name}_{idx}",
                    }
                    data.append(preprocessed)

        with open(self.qasc.validate_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact1 = entry["fact1"].strip(".").lower()
                fact2 = entry["fact2"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "choices": [c["text"] for c in entry["question"]["choices"]],
                    "facts": [fact1, fact2],
                    "id": f"qasc_validate_{idx}",
                }
                data.append(preprocessed)
        return data

    def generate_facts(self):
        facts = set()
        for path in (
            self.qasc.train_path,
            self.qasc.validate_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    entry = json.loads(line)
                    facts.add(entry["fact1"].strip(".").lower())
                    facts.add(entry["fact2"].strip(".").lower())
        for path in (
            self.openbook_qa.facts_path,
            self.openbook_qa.crowd_source_facts_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    line = line.strip("\n").strip('"').strip(".")
                    if len(line) < 3:
                        continue
                    facts.add(line.lower())
        logging.info(f"Generated {len(facts)} facts")
        return list(facts)

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "facts": self.facts,
                    "train_retriever": self.train_retriever_data,
                    "train_reranker": self.train_reranker_data,
                    "unannotated_train": self.unannotated_train_data,
                    "search": self.search_data,
                    "validate": self.validate_data,
                },
                file,
            )

    def annotate_train_data(
        self, batch: BatchEncoding, choice: t.Tensor, confidence: t.Tensor
    ):
        for i in range(choice.shape[0]):
            if confidence[i].item():
                fact = self.facts[choice[i].item()]
                annotated_data, idx = [
                    (d, idx)
                    for idx, d in enumerate(self.unannotated_train_data)
                    if d["id"] == batch["id"][i]
                ][0]
                annotated_data["facts"] = [fact]
                new_data = self.unannotated_train_data.pop(idx)
                self.train_retriever_data.append(new_data)
                self.train_reranker_data.append(new_data)
        logging.info(f"Annotated {confidence.to(dtype=t.int32).sum()} train_data")

    def set_train_candidate_fact_indices(
        self, batch: BatchEncoding, candidates: t.Tensor
    ):
        # Note, use the length of the retriever dataset here since reranker dataset
        # contains pseudo data without a real fact
        if candidates.shape[0] != len(self.train_retriever_data):
            raise ValueError(
                f"Candidate size {candidates.shape[0]} does not match "
                f"train dataset size {len(self.train_retriever_data)}"
            )
        # The actual annotated data comes from the reranker dataset
        for i in range(candidates.shape[0]):
            annotated_data = [
                d for d in self.train_reranker_data if d["id"] == batch["id"][i]
            ][0]
            annotated_data["candidate_fact_indices"] = candidates[i].cpu().tolist()

    def save_search_targets(self, choice: t.Tensor):
        target_dict = {}
        if choice.shape[0] != len(self.search_data):
            raise ValueError(
                f"Choice size {choice.shape[0]} does not match "
                f"search target size {len(self.search_data)}"
            )
        for i in range(choice.shape[0]):
            if choice.dim() == 1:
                target_dict[self.search_data[i]["id"]] = self.facts[choice[i].item()]
            else:
                target_dict[self.search_data[i]["id"]] = [
                    self.facts[c.item()] for c in choice[i]
                ]
        with open(
            os.path.join(preprocess_cache_dir, f"arc_targets.json"), "w",
        ) as file:
            json.dump(target_dict, file)
