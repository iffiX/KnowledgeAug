import os
import copy
import json
import pickle
import difflib
import logging
import nltk
import numpy as np
import torch as t
from typing import List
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.download import CommonsenseQA
from encoder.dataset.matcher.commonsense_qa import CommonsenseQAMatcher
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.file import open_file_with_create_directories
from encoder.utils.inspect import save_inspect_data
from .base import StaticIterableDataset


class CommonsenseQADataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 300,
        generate_length: int = 32,
        use_matcher: bool = False,
        matcher_mode: str = "embedding",
        matcher_seed: int = -1,
        matcher_config: dict = None,
        include_option_label_in_sentence: bool = False,
        include_option_label_in_answer_and_choices: bool = False,
        use_option_label_as_answer_and_choices: bool = False,
        insert_answers_at_end: bool = False,
        match_closest_when_no_equal: bool = True,
        regenerate: bool = True,
        output_mode: str = "single",
    ):
        self.tokenizer = tokenizer
        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_seq_length = max_seq_length
        self.generate_length = generate_length
        self.use_matcher = use_matcher
        self.matcher_mode = matcher_mode
        self.matcher_seed = matcher_seed
        self.matcher_config = matcher_config
        self.include_option_label_in_sentence = include_option_label_in_sentence
        self.include_option_label_in_answer_and_choices = (
            include_option_label_in_answer_and_choices
        )
        self.insert_answers_at_end = insert_answers_at_end
        self.use_option_label_as_answer_and_choices = (
            use_option_label_as_answer_and_choices
        )
        self.match_closest_when_no_equal = match_closest_when_no_equal
        self.regenerate = regenerate

        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")
        self.output_mode = output_mode
        self.question_matcher = CommonsenseQAMatcher(
            tokenizer=self.matcher_tokenizer, for_question_annotation=True
        )
        self.matcher = CommonsenseQAMatcher(tokenizer=self.matcher_tokenizer)
        self.commonsense_qa = CommonsenseQA().require()

        archive_path = os.path.join(preprocess_cache_dir, "commonsense_qa.data")
        if not os.path.exists(archive_path):
            self.train_data = self.parse_data(self.commonsense_qa.train_path)
            self.validate_data = self.parse_data(self.commonsense_qa.validate_path)
            self.test_data = self.parse_data(self.commonsense_qa.test_path)
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
            if (
                data["include_option_label_in_answer_and_choices"]
                != self.include_option_label_in_answer_and_choices
                or data["include_option_label_in_sentence"]
                != self.include_option_label_in_sentence
                or data["use_option_label_as_answer_and_choices"]
                != self.use_option_label_as_answer_and_choices
            ):
                if regenerate:
                    logging.info(
                        "Configuration mismatch, regenerating commonsense qa dataset."
                    )
                    self.train_data = self.parse_data(self.commonsense_qa.train_path)
                    self.validate_data = self.parse_data(
                        self.commonsense_qa.validate_path
                    )
                    self.test_data = self.parse_data(self.commonsense_qa.test_path)
                    self.save(archive_path)
                else:
                    raise ValueError("Configuration mismatch")
            else:
                self.train_data = data["train"]
                self.validate_data = data["validate"]
                self.test_data = data["test"]

        self.disable_dict = {"train": [], "validate": []}
        for split, dataset in (
            ("train", self.train_data),
            ("validate", self.validate_data),
        ):
            for data in dataset:
                try:
                    nodes = self.matcher.matcher.kb.find_nodes(
                        [data["text_question"] + " " + data["choices"][data["label"]]]
                    )
                except ValueError:
                    nodes = None
                self.disable_dict[split].append(nodes)

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
        if split == "train":
            data = self.train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        else:
            data = self.test_data[index]
        if self.output_mode == "single":
            if self.use_matcher:
                # prevent any modification to data, also prevent checkpoint storing
                # data to gpu by moving
                data = copy.deepcopy(data)
                if self.matcher_mode == "embedding":
                    wnl = WordNetLemmatizer()

                    if len(data["target"]) > 0:
                        # For supporting manually setting search targets
                        allowed_tokens = data["target"]
                    else:
                        tokens = nltk.word_tokenize(data["text_question"])
                        allowed_tokens = [data["question_concept"]]
                        tagged = nltk.pos_tag(tokens)
                        for token, pos in tagged:
                            if pos.startswith("NN"):
                                allowed_tokens.append(wnl.lemmatize(token))
                        if len(allowed_tokens) < 3:
                            for token, pos in tagged:
                                if pos.startswith("JJ"):
                                    allowed_tokens.append(wnl.lemmatize(token))

                    target = " @ ".join(sorted(list(set(allowed_tokens))))
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
                        disabled_nodes=self.disable_dict[split][index]
                        if split in ("train", "validate")
                        else None,
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

                    new_question = self.matcher.insert_selection(
                        data["text_question"], selection, insert_at_end=True, sep="#"
                    )

                    choice_mask = "+" * len(data["text_choices"])
                    for choice in ("[A]", "[B]", "[C]", "[D]", "[E]"):
                        start_pos = data["text_choices"].find(choice)
                        if start_pos != -1:
                            choice_mask = (
                                choice_mask[:start_pos]
                                + "-" * len(choice)
                                + choice_mask[start_pos + len(choice) :]
                            )

                    match = self.matcher.match_by_node_embedding(
                        data["text_choices"],
                        target_sentence=target,
                        target_mask=target_mask,
                        source_mask=choice_mask,
                        disabled_nodes=self.disable_dict[split][index]
                        if split in ("train", "validate")
                        else None,
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

                    new_choices = self.matcher.insert_selection(
                        data["text_choices"], selection, insert_at_end=True, sep="#"
                    )
                    for choice in ("a", "b", "c", "d"):
                        new_choices = new_choices.replace(
                            f"[ {choice} ]", f"[{choice.upper()}]"
                        )
                elif self.matcher_mode == "none":
                    new_question = data["text_question"]
                    new_choices = data["text_choices"]
                else:
                    raise ValueError(f"Invalid match mode {self.matcher_mode}")

                encoded_sentence = self.tokenizer(
                    new_question,
                    new_choices,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            else:
                encoded_sentence = self.tokenizer(
                    data["text_question"],
                    data["text_choices"],
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            answer = self.tokenizer.encode(
                data["text_answer"],
                padding="max_length",
                max_length=self.generate_length,
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
                    wnl = WordNetLemmatizer()

                    if len(data["target"]) > 0:
                        # For supporting manually setting search targets
                        question_allowed_tokens = data["target"]
                    else:
                        right_choice = data["choices"][data["label"]]
                        wrong_choices = list(
                            set(data["choices"]).difference({right_choice})
                        )
                        question_allowed_tokens = [
                            right_choice,
                            self.question_matcher.find_closest_concept(
                                right_choice, wrong_choices
                            ),
                        ]

                    question_target = " @ ".join(
                        sorted(list(set(question_allowed_tokens)))
                    )
                    question_target_mask = []
                    for c in question_target:
                        if c == "@":
                            question_target_mask.append("-")
                        else:
                            question_target_mask.append("+")
                    question_target_mask = "".join(question_target_mask)

                    tokens = nltk.word_tokenize(data["text_question"])
                    choice_allowed_tokens = [data["question_concept"]]
                    tagged = nltk.pos_tag(tokens)
                    for token, pos in tagged:
                        if pos.startswith("NN"):
                            choice_allowed_tokens.append(wnl.lemmatize(token))
                    if len(choice_allowed_tokens) < 3:
                        for token, pos in tagged:
                            if pos.startswith("JJ"):
                                choice_allowed_tokens.append(wnl.lemmatize(token))

                    choice_target = " @ ".join(sorted(list(set(choice_allowed_tokens))))
                    choice_target_mask = []
                    for c in choice_target:
                        if c == "@":
                            choice_target_mask.append("-")
                        else:
                            choice_target_mask.append("+")
                    choice_target_mask = "".join(choice_target_mask)

                    # new_question = data["text_question"]

                    # new_question = self.insert_choice_by_weight_to_question(
                    #     data["text_question"], data["question_concept"], data["choices"]
                    # )

                    match = self.question_matcher.match_by_node_embedding(
                        data["text_question"],
                        target_sentence=question_target,
                        target_mask=question_target_mask,
                        seed=self.matcher_seed,
                        max_times=self.matcher_config["question_match_max_times"],
                        max_depth=self.matcher_config["question_match_max_depth"],
                        edge_top_k=self.matcher_config["question_match_edge_top_k"],
                        split_node_minimum_edge_num=0,
                        source_context_range=self.matcher_config[
                            "question_match_source_context_range"
                        ],
                    )
                    selection = self.question_matcher.select_paths(
                        match,
                        max_edges=self.matcher_config["question_select_max_edges"],
                        discard_edges_if_rank_below=self.matcher_config[
                            "question_select_discard_edges_if_rank_below"
                        ],
                        filter_short_accurate_paths=False,
                    )

                    new_question = self.question_matcher.insert_selection(
                        data["text_question"], selection, insert_at_end=True, sep=",",
                    )

                    new_choices = []
                    # new_choices = data["choices"]
                    for choice in data["choices"]:
                        match = self.matcher.match_by_node_embedding(
                            choice,
                            target_sentence=choice_target,
                            target_mask=choice_target_mask,
                            disabled_nodes=self.disable_dict[split][index]
                            if split in ("train", "validate")
                            else None,
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
                            self.matcher.insert_selection(
                                choice, selection, insert_at_end=True
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

    def insert_choice_by_weight_to_question(self, question, question_concept, choices):
        kb = self.matcher.matcher.kb
        qc_node = kb.find_nodes([question_concept])[0]
        weights = []
        choice_edge_names = []
        for choice in choices:
            try:
                ch_node = kb.find_nodes([choice])[0]
            except ValueError:
                weights.append(-1)
                choice_edge_names.append(None)
            else:
                edges = kb.get_edges(qc_node, ch_node)
                if len(edges) > 0:
                    weights.append(edges[0][3])
                    choice_edge_names.append(
                        f"{kb.nodes[edges[0][0]]} "
                        f"{kb.relationships[edges[0][1]]} "
                        f"{kb.nodes[edges[0][2]]}"
                    )
                else:
                    weights.append(-1)
                    choice_edge_names.append(None)
        idx = np.argmax(weights)
        if weights[idx] != -1:
            return question + f" ( {choice_edge_names[idx]} # )"
        else:
            return question

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

        save_inspect_data(answers, "commensense_qa_val_answers")
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

    def set_search_target(self, search_target: List[str], split: str, id):
        if split == "validate":
            split_data = self.validate_data
        elif split == "test":
            split_data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        found_data = [d for d in split_data if d["id"] == id]
        if len(found_data) != 1:
            raise ValueError(f"Id {id} not found in split {split}")
        # prevent raising an exception since sometimes the target may be empty
        if len(search_target) == 0:
            search_target.append("")

        found_data[0]["target"] = search_target

    def parse_data(self, path):
        data = []
        logging.info(f"Parsing {path}")
        with open_file_with_create_directories(path, "r") as file:
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
                    "target": [],
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
                    preprocessed["text_answer"] = self.generate_text_answer(
                        label, choices[label]
                    )

                data.append(preprocessed)
        return data

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "train": self.train_data,
                    "validate": self.validate_data,
                    "test": self.test_data,
                    "include_option_label_in_sentence": self.include_option_label_in_sentence,
                    "include_option_label_in_answer_and_choices": self.include_option_label_in_answer_and_choices,
                    "use_option_label_as_answer_and_choices": self.use_option_label_as_answer_and_choices,
                },
                file,
            )

    def generate_choice_str(self, choices: List[str]):
        if self.include_option_label_in_sentence:
            result = ""
            options = ["[A]", "[B]", "[C]", "[D]", "[E]"]
            for option, choice in zip(options, choices):
                result += option + " " + choice + " "
            return result
        else:
            return ", ".join(choices)

    def generate_text_answer(self, label, choice):
        options = ["[A]", "[B]", "[C]", "[D]", "[E]"]
        if self.include_option_label_in_answer_and_choices:
            text_answer = f"{options[label]} {choice.lower().strip(',')}"
        elif self.use_option_label_as_answer_and_choices:
            text_answer = f"{options[label]}"
        else:
            text_answer = choice.lower().strip(",")
        return text_answer
