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


class CommonsenseQA2Matcher(BaseMatcher):
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

        super(CommonsenseQA2Matcher, self).__init__(tokenizer, matcher)
        self.matcher.kb.disable_edges_with_weight_below(1)
        self.allowed_composite_nodes = {}
        self.allowed_facts = {}

    def add_external_knowledge(
        self, external_knowledge: List[str], allowed_ids: List[str]
    ):
        logging.info("Adding external knowledge")

        def generate():
            added = set()
            for knowledge in external_knowledge:
                knowledge = knowledge.strip(".").replace('"', " ").lower()
                if knowledge not in added:
                    added.add(knowledge)
            added_knowledge = list(sorted(added))
            tokens = [
                (knowledge, token_ids, token_mask)
                for knowledge, (token_ids, token_mask) in zip(
                    added_knowledge, self.parallel_tokenize_and_mask(added_knowledge)
                )
            ]
            return hash("".join(external_knowledge)), tokens

        def validate(data):
            return hash("".join(external_knowledge)) == data[0]

        with PickleCache(
            os.path.join(
                preprocess_cache_dir, "commonsense_qa2_matcher_external_knowledge.data"
            ),
            generate,
            validate_func=validate,
        ) as cache:
            hash_value, tokens = cache.data
            for allowed_id, (knowledge, token_ids, token_mask) in tqdm(
                zip(allowed_ids, tokens)
            ):
                node_id = self.matcher.kb.add_composite_node(
                    knowledge, "RelatedTo", token_ids, token_mask
                )
                if allowed_id not in self.allowed_composite_nodes:
                    self.allowed_composite_nodes[allowed_id] = []
                    self.allowed_facts[allowed_id] = []
                self.allowed_composite_nodes[allowed_id].append(node_id)
                self.allowed_facts[allowed_id].append(knowledge)
            logging.info(f"Added {len(tokens)} composite nodes")

    def __reduce__(self):
        return (
            CommonsenseQA2Matcher,
            (self.tokenizer,),
        )
