import re
import os
import json
import logging
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNetWithGloVe, OpenBookQA
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.settings import preprocess_cache_dir


class OpenBookQAMatcher(BaseMatcher):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive-glove.data")
        )
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding-glove.hdf5")
        )
        self.concept_net = ConceptNetWithGloVe().require()
        self.openbook_qa = OpenBookQA().require()

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
            logging.info("Initializing KnowledgeMatcher")
            matcher = KnowledgeMatcher(archive_path)
        super(OpenBookQAMatcher, self).__init__(tokenizer, matcher)

        self.matcher.kb.disable_edges_with_weight_below(1)

        # Add knowledge from openbook QA as composite nodes
        self.add_openbook_qa_knowledge()
        # self.add_openbook_qa_train_dataset()

    def add_openbook_qa_knowledge(self):
        logging.info("Adding OpenBook QA knowledge")
        # qasc_additional_path = os.path.join(preprocess_cache_dir, "qasc_additional.txt")
        manual_additional_path = os.path.join(
            preprocess_cache_dir, "manual_additional.txt"
        )
        count = 0
        for path in (
            self.openbook_qa.facts_path,
            self.openbook_qa.crowd_source_facts_path,
            # qasc_additional_path,
            manual_additional_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    line = line.strip("\n").strip(".").strip('"').strip("'").strip(",")
                    if len(line) < 3:
                        continue
                    count += 1
                    ids, mask = self.tokenize_and_mask(line)
                    self.matcher.kb.add_composite_node(line, "RelatedTo", ids, mask)
        logging.info(f"Added {count} composite nodes")

    def add_openbook_qa_train_dataset(self):
        logging.info("Adding OpenBook QA train dataset")
        count = 0
        with open(self.openbook_qa.train_path, "r") as file:
            for line in file:
                sample = json.loads(line)
                correct_choice = [
                    c["text"]
                    for c in sample["question"]["choices"]
                    if c["label"] == sample["answerKey"]
                ][0]
                line = sample["question"]["stem"] + " " + correct_choice
                count += 1
                ids, mask = self.tokenize_and_mask(line)
                self.matcher.kb.add_composite_node(line, "RelatedTo", ids, mask)
        logging.info(f"Added {count} composite nodes")

    def __reduce__(self):
        return OpenBookQAMatcher, (self.tokenizer,)
