import re
import os
import logging
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

        super(CommonsenseQA2Matcher, self).__init__(tokenizer, matcher, archive_path)
        self.matcher.kb.disable_edges_with_weight_below(1)

    def add_external_knowledge(self, external_knowledge: List[str]):
        logging.info("Adding external knowledge")

        def generate():
            added_knowledge = sorted(list(set(external_knowledge)))
            tokens = [
                (knowledge, token_ids, token_mask)
                for knowledge, (token_ids, token_mask) in zip(
                    added_knowledge, self.parallel_tokenize_and_mask(added_knowledge)
                )
            ]
            return len(external_knowledge), tokens

        def validate(data):
            return len(external_knowledge) == data[0]

        with PickleCache(
            os.path.join(
                preprocess_cache_dir, "commonsense_qa2_matcher_external_knowledge.data"
            ),
            generate,
            validate_func=validate,
        ) as cache:
            hash_value, tokens = cache.data
            for knowledge, token_ids, token_mask in tqdm(tokens):
                self.matcher.kb.add_composite_node(
                    knowledge, "RelatedTo", token_ids, token_mask
                )
            logging.info(f"Added {len(tokens)} composite nodes")

    def __reduce__(self):
        return (
            CommonsenseQA2Matcher,
            (self.tokenizer,),
        )
