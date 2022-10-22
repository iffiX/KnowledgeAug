import re
import os
import json
import logging
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNetWithGloVe, OpenBookQA
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.file import PickleCache
from encoder.utils.settings import preprocess_cache_dir
from .fact_selector import FactSelector


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
        super(OpenBookQAMatcher, self).__init__(tokenizer, matcher, archive_path)

        self.matcher.kb.disable_edges_with_weight_below(1)

        self.allowed_composite_nodes = {}
        self.fact_to_composite_node = {}
        self.composite_node_to_fact = {}

        # Add knowledge from openbook QA as composite nodes
        self.add_openbook_qa_knowledge()
        # self.add_openbook_qa_train_dataset()

    def add_openbook_qa_knowledge(self):
        logging.info("Adding OpenBook QA knowledge")
        count = 0

        def generate_filtered_openbook_qa_corpus_tokens():
            questions_with_choices = []
            question_ids = []
            for dataset_path in (
                self.openbook_qa.train_path,
                self.openbook_qa.validate_path,
                self.openbook_qa.test_path,
            ):
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        question_ids.append(entry["id"])
                        questions_with_choices.append(entry["question"]["stem"])
                        for choice in entry["question"]["choices"]:
                            questions_with_choices.append(
                                entry["question"]["stem"] + " " + choice["text"]
                            )

            raw_facts = []
            for path in (
                self.openbook_qa.facts_path,
                self.openbook_qa.crowd_source_facts_path,
                self.openbook_qa.search_result_path,
            ):
                with open(path, "r") as file:
                    for line in file:
                        line = (
                            line.strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                        )
                        if len(line) >= 3:
                            raw_facts.append(line)

            fact_selector = FactSelector(
                questions_with_choices, raw_facts, min_score=0.1, max_facts=150
            )
            selected_facts = []
            for start in tqdm(list(range(0, len(questions_with_choices), 5))):
                per_question_facts = []
                for i in range(5):
                    per_question_facts += [
                        fact for fact in fact_selector.selected_facts[start + i]
                    ]
                selected_facts.append(per_question_facts)

            tokens = [
                (fact, ids, mask)
                for fact, (ids, mask) in zip(
                    raw_facts, self.parallel_tokenize_and_mask(raw_facts),
                )
            ]
            return question_ids, selected_facts, tokens

        with PickleCache(
            os.path.join(
                preprocess_cache_dir, "openbook_qa_matcher_filtered_corpus.data"
            ),
            generate_filtered_openbook_qa_corpus_tokens,
        ) as cache:
            question_ids, list_of_question_allowed_facts, tokens = cache.data
            for knowledge, ids, mask in tqdm(tokens):
                self.fact_to_composite_node[
                    knowledge
                ] = self.matcher.kb.add_composite_node(
                    knowledge, "RelatedTo", ids, mask
                )
            for question_id, question_allowed_facts in zip(
                question_ids, list_of_question_allowed_facts
            ):
                self.allowed_composite_nodes[question_id] = [
                    self.fact_to_composite_node[f] for f in question_allowed_facts
                ]
            for fact, node in self.fact_to_composite_node.items():
                self.composite_node_to_fact[node] = fact
            logging.info(f"Added {len(tokens)} composite nodes")

    def __reduce__(self):
        return OpenBookQAMatcher, (self.tokenizer,)
