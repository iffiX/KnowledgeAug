import re
import os
import copy
import json
import random
import logging
import datasets
from tqdm import tqdm
from typing import List
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNet, CommonsenseQA
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.file import PickleCache
from encoder.utils.settings import preprocess_cache_dir, dataset_cache_dir


class CommonsenseQAMatcher(BaseMatcher):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase,
    ):
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
        )
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding-glove.hdf5")
        )
        self.concept_net = ConceptNet().require()
        self.commonsense_qa = CommonsenseQA().require()

        if not os.path.exists(archive_path):
            logging.info("Processing concept net")
            reader = ConceptNetReader().read(
                asserion_path=self.concept_net.assertion_path,
                weight_path=self.concept_net.glove_path,
                weight_style="glove_42b_300d",
                weight_hdf5_path=embedding_path,
                simplify_with_int8=True,
            )
            reader.tokenized_nodes = tokenizer(
                reader.nodes, add_special_tokens=False
            ).input_ids
            relationships = [
                " ".join([string.lower() for string in re.findall("[A-Z][a-z]*", rel)])
                for rel in reader.relationships
            ]
            reader.tokenized_relationships = tokenizer(
                relationships, add_special_tokens=False
            ).input_ids
            reader.tokenized_edge_annotations = tokenizer(
                [edge[4] for edge in reader.edges], add_special_tokens=False
            ).input_ids
            matcher = KnowledgeMatcher(reader)
            logging.info("Saving preprocessed concept net data as archive")
            matcher.save(archive_path)
        else:
            matcher = KnowledgeMatcher(archive_path)

        super(CommonsenseQAMatcher, self).__init__(tokenizer, matcher, archive_path)
        self.matcher.kb.disable_edges_with_weight_below(1)

        self.add_generics_kb()
        self.add_commonsense_qa_dataset()
        # self.add_openbook_qa_knowledge()

    def add_question_specific_knowledge(self, question_specific_knowledge: List[str]):
        logging.info("Adding question specific knowledge")
        added = set()
        for knowledge in question_specific_knowledge:
            knowledge = (
                knowledge.strip(".")
                .replace("(", ",")
                .replace(")", ",")
                .replace(";", ",")
                .replace('"', " ")
                .lower()
            )
            if knowledge not in added:
                added.add(knowledge)
                ids, mask = self.tokenize_and_mask(knowledge)
                self.matcher.kb.add_composite_node(knowledge, "RelatedTo", ids, mask)
        logging.info(f"Added {len(added)} composite nodes")

    def add_commonsense_qa_dataset(self):
        logging.info("Adding Commonsense QA dataset")
        count = 0
        for dataset_path in (
            self.commonsense_qa.train_path,
            # self.commonsense_qa.validate_path,
        ):
            with open(dataset_path, "r") as file:
                for line in file:
                    sample = json.loads(line)
                    correct_choice = [
                        c["text"]
                        for c in sample["question"]["choices"]
                        if c["label"] == sample["answerKey"]
                    ][0]
                    line = sample["question"]["stem"] + " " + correct_choice
                    if line.count(".") >= 3:
                        continue
                    count += 1
                    ids, mask = self.tokenize_and_mask(line)
                    self.matcher.kb.add_composite_node(line, "RelatedTo", ids, mask)
        logging.info(f"Added {count} composite nodes")

    def add_generics_kb(self):
        logging.info("Adding generics kb")
        path = str(os.path.join(dataset_cache_dir, "generics_kb"))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(path, exist_ok=True)
        if not os.path.exists(
            os.path.join(path, "GenericsKB-Best.tsv")
        ) and not os.path.isfile(path):
            logging.info("Skipping loading generics_kb because file is not loaded")
            logging.info(
                f"Please download GenericsKB-Best.tsv "
                f"from https://drive.google.com/drive/folders/1vqfVXhJXJWuiiXbUa4rZjOgQoJvwZUoT "
                f"to path {os.path.join(path, 'GenericsKB-Best.tsv')}"
            )
            return

        def generate():
            added = set()
            logging.info(f"Preprocessing generics_kb")
            gkb = datasets.load_dataset(
                "generics_kb", "generics_kb_best", data_dir=path,
            )
            for entry in tqdm(gkb["train"]):
                if (
                    "ConceptNet" in entry["source"]
                    or "WordNet" in entry["source"]
                    or "TupleKB" in entry["source"]
                    or entry["generic_sentence"].count(" ") < 3
                ):
                    continue
                knowledge = (
                    entry["generic_sentence"]
                    .strip(".")
                    .replace("(", ",")
                    .replace(")", ",")
                    .replace(";", ",")
                    .replace('"', " ")
                    .lower()
                )
                if knowledge not in added:
                    added.add(knowledge)
            added = list(added)
            result = [
                (knowledge, ids, mask)
                for knowledge, (ids, mask) in zip(
                    added, self.parallel_tokenize_and_mask(added)
                )
            ]
            return result

        with PickleCache(
            os.path.join(
                preprocess_cache_dir, "commonsense_qa_generics_kb_knowledge.data"
            ),
            generate_func=generate,
        ) as cache:
            # rand = random.Random(42)
            # to_be_added = copy.deepcopy(cache.data)
            # rand.shuffle(to_be_added)
            # to_be_added = to_be_added[:100000]
            to_be_added = copy.deepcopy(cache.data)
            for knowledge, ids, mask in tqdm(to_be_added):
                self.matcher.kb.add_composite_node(knowledge, "RelatedTo", ids, mask)
            logging.info(f"Added {len(to_be_added)} composite nodes")

    # def add_openbook_qa_knowledge(self):
    #     logging.info("Adding OpenBook QA knowledge")
    #     openbook_qa_path = os.path.join(
    #         dataset_cache_dir, "openbook_qa", "OpenBookQA-V1-Sep2018", "Data"
    #     )
    #     openbook_qa_facts_path = os.path.join(openbook_qa_path, "Main", "openbook.txt")
    #     crowd_source_facts_path = os.path.join(
    #         openbook_qa_path, "Additional", "crowdsourced-facts.txt"
    #     )
    #     qasc_additional_path = os.path.join(preprocess_cache_dir, "qasc_additional.txt")
    #     manual_additional_path = os.path.join(
    #         preprocess_cache_dir, "manual_additional.txt"
    #     )
    #     count = 0
    #     for path in (
    #         openbook_qa_facts_path,
    #         crowd_source_facts_path,
    #         # qasc_additional_path,
    #         manual_additional_path,
    #     ):
    #         with open(path, "r") as file:
    #             for line in file:
    #                 line = line.strip("\n").strip(".").strip('"').strip("'").strip(",")
    #                 if len(line) < 3:
    #                     continue
    #                 count += 1
    #                 ids, mask = self.tokenize_and_mask(line)
    #                 self.matcher.kb.add_composite_node(line, "RelatedTo", ids, mask)
    #     logging.info(f"Added {count} composite nodes")

    #

    def __reduce__(self):
        return (
            CommonsenseQAMatcher,
            (self.tokenizer,),
        )
