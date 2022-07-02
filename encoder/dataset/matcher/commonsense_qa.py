import re
import os
import json
import logging
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNet, CommonsenseQA
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.settings import preprocess_cache_dir


class CommonsenseQAMatcher(BaseMatcher):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, for_question_annotation=False
    ):
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
        )
        self.concept_net = ConceptNet().require()
        self.commonsense_qa = CommonsenseQA().require()

        if not os.path.exists(archive_path):
            logging.info("Processing concept net")
            reader = ConceptNetReader().read(
                asserion_path=self.concept_net.assertion_path,
                weight_path=self.concept_net.numberbatch_path,
                weight_hdf5_path=self.concept_net.embedding_path,
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
        super(CommonsenseQAMatcher, self).__init__(tokenizer, matcher)

        if not for_question_annotation:
            # Disable relations of similar word forms
            matcher.kb.disable_edges_of_relationships(
                [
                    "DerivedFrom",
                    "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo",
                    "RelatedTo",
                    "FormOf",
                ]
            )
        else:
            matcher.kb.disable_edges_of_relationships(
                [
                    "DerivedFrom",
                    "EtymologicallyDerivedFrom",
                    "EtymologicallyRelatedTo",
                    "RelatedTo",
                    "FormOf",
                    "DefinedAs",
                    # "IsA",
                    # "MannerOf",
                ]
            )
        self.matcher.kb.disable_edges_with_weight_below(1)
        super(CommonsenseQAMatcher, self).__init__(tokenizer, matcher)

        if not for_question_annotation:
            self.add_commonsense_qa_dataset()
            # self.add_wordnet_definition()
            # self.add_openbook_qa_knowledge()
            # self.add_generics_kb()

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

    # def add_generics_kb(self):
    #     logging.info("Adding generics kb")
    #     path = str(os.path.join(dataset_cache_dir, "generics_kb"))
    #     os.makedirs(path, exist_ok=True)
    #     if not os.path.exists(os.path.join(path, "GenericsKB-Best.tsv")):
    #         logging.info("Skipping loading generics_kb because file is not loaded")
    #         logging.info(
    #             f"Please download GenericsKB-Best.tsv "
    #             f"from https://drive.google.com/drive/folders/1vqfVXhJXJWuiiXbUa4rZjOgQoJvwZUoT "
    #             f"to path {os.path.join(path, 'GenericsKB-Best.tsv')}"
    #         )
    #     gkb = datasets.load_dataset("generics_kb", "generics_kb_best", data_dir=path,)
    #     added = set()
    #     for entry in gkb["train"]:
    #         if (
    #             "ConceptNet" in entry["source"]
    #             or "WordNet" in entry["source"]
    #             or entry["generic_sentence"].count(" ") < 3
    #         ):
    #             continue
    #         knowledge = (
    #             entry["generic_sentence"]
    #             .strip(".")
    #             .replace("(", ",")
    #             .replace(")", ",")
    #             .replace(";", ",")
    #             .replace('"', " ")
    #             .lower()
    #         )
    #         if knowledge not in added:
    #             added.add(knowledge)
    #             self.matcher.kb.add_composite_node(
    #                 knowledge,
    #                 "RelatedTo",
    #                 self.tokenizer.encode(knowledge, add_special_tokens=False),
    #             )
    #     logging.info(f"Added {len(added)} composite nodes")

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
        return CommonsenseQAMatcher, (self.tokenizer,)
