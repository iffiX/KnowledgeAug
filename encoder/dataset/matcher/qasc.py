import re
import os
import json
import pickle
import logging
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNetWithGloVe, OpenBookQA, QASC
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.file import JSONCache, PickleCache
from encoder.utils.settings import preprocess_cache_dir
from .fact_selector import FactSelector


class QASCMatcher(BaseMatcher):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive-glove.data")
        )
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding-glove.hdf5")
        )
        self.concept_net = ConceptNetWithGloVe().require()
        self.openbook_qa = OpenBookQA().require()
        self.qasc = QASC().require()

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
        super(QASCMatcher, self).__init__(tokenizer, matcher)

        self.matcher.kb.disable_edges_with_weight_below(1)
        self.matcher.kb.disable_edges_of_relationships(["HasContext"])

        self.added_qasc_corpus_facts = None
        self.allowed_composite_nodes = {}
        self.fact_to_composite_node = {}
        self.composite_node_to_fact = {}
        self.add_qasc_corpus()

    def add_qasc_facts(self, train_and_validate=False):
        logging.info(
            f"Adding QASC facts ({'train and validate' if train_and_validate else 'train'})"
        )

        def generate():
            added = set()
            paths = (
                [self.qasc.train_path]
                if not train_and_validate
                else [self.qasc.train_path, self.qasc.validate_path,]
            )
            question_facts = {}
            for dataset_path in paths:
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        fact1 = (
                            entry["fact1"]
                            .strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                        )
                        fact2 = (
                            entry["fact2"]
                            .strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                        )
                        added.add(fact1)
                        added.add(fact2)
                        question_facts[entry["id"]] = (fact1, fact2)
            knowledge = list(added)
            tokens = [
                (fact, ids, mask)
                for fact, (ids, mask) in zip(
                    knowledge, self.parallel_tokenize_and_mask(knowledge)
                )
            ]
            return question_facts, tokens

        count = 0
        with PickleCache(
            os.path.join(
                preprocess_cache_dir,
                f"qasc_matcher_{'train_val' if train_and_validate else 'train'}_facts_knowledge.data",
            ),
            generate,
        ) as cache:
            question_facts, tokens = cache.data

            for knowledge, ids, mask in tqdm(tokens):
                if knowledge not in self.added_qasc_corpus_facts:
                    count += 1
                    self.fact_to_composite_node[
                        knowledge
                    ] = self.matcher.kb.add_composite_node(
                        knowledge, "RelatedTo", ids, mask
                    )

            for question_id, question_allowed_facts in question_facts.items():
                self.allowed_composite_nodes[question_id] = list(
                    set(
                        self.allowed_composite_nodes[question_id]
                        + [
                            self.fact_to_composite_node[f]
                            for f in question_allowed_facts
                        ]
                    )
                )
        logging.info(f"Added {count} composite nodes")

    def add_qasc_corpus(self):
        logging.info("Adding QASC corpus")

        if not os.path.exists(
            os.path.join(preprocess_cache_dir, "qasc_matcher_filtered_corpus.data")
        ):

            questions = []
            questions_with_choices = []
            question_ids = []
            for dataset_path in (
                self.qasc.train_path,
                self.qasc.validate_path,
                self.qasc.test_path,
            ):
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        question_ids.append(entry["id"])
                        questions.append(entry["question"]["stem"])
                        questions_with_choices.append(entry["question"]["stem"])
                        for choice in entry["question"]["choices"]:
                            questions_with_choices.append(
                                entry["question"]["stem"] + " " + choice["text"]
                            )

            def generate_first_level_facts():
                with open(self.qasc.corpus_path, "r") as file:
                    raw_facts = [line.strip("\n") for line in file]
                fact_selector = FactSelector(
                    questions, raw_facts, min_score=0.5, max_facts=100
                )
                return {
                    "queries": questions,
                    "query_facts": fact_selector.selected_facts,
                    "query_facts_rank": fact_selector.selected_facts_rank,
                }

            with JSONCache(
                os.path.join(preprocess_cache_dir, "qasc_matcher_facts.json"),
                generate_first_level_facts,
            ) as cache:
                facts = sorted(
                    list(
                        set(
                            [
                                f.strip("\n")
                                .strip(".")
                                .strip('"')
                                .strip("'")
                                .strip(",")
                                .lower()
                                for facts in cache.data["query_facts"]
                                for f in facts[:100]
                            ]
                        )
                    )
                )

            def generate_filtered_qasc_corpus_tokens():
                fact_selector = FactSelector(
                    questions_with_choices,
                    facts,
                    min_score=0.4,
                    max_facts=1000,
                    inner_batch_size=4096,
                )
                # Select top 5000 facts for each question
                selected_facts = []
                for start in tqdm(list(range(0, len(questions_with_choices), 9))):
                    facts_ranks = []
                    for i in range(9):
                        facts_ranks += fact_selector.selected_facts_rank[start + i]
                    facts_ranks = sorted(facts_ranks, key=lambda x: x[1], reverse=True)[
                        :5000
                    ]
                    selected_facts.append([facts[rank[0]] for rank in facts_ranks])
                tokens = [
                    (fact, ids, mask)
                    for fact, (ids, mask) in zip(
                        facts, self.parallel_tokenize_and_mask(facts),
                    )
                ]
                return question_ids, selected_facts, tokens

            with PickleCache(
                os.path.join(preprocess_cache_dir, "qasc_matcher_filtered_corpus.data"),
                generate_filtered_qasc_corpus_tokens,
            ) as cache:
                filtered_corpus = cache.data
        else:
            with open(
                os.path.join(preprocess_cache_dir, "qasc_matcher_filtered_corpus.data"),
                "rb",
            ) as file:
                filtered_corpus = pickle.load(file)

        question_ids, list_of_question_allowed_facts, tokens = filtered_corpus

        self.added_qasc_corpus_facts = set()
        for knowledge, ids, mask in tqdm(tokens):
            self.added_qasc_corpus_facts.add(knowledge)
            self.fact_to_composite_node[knowledge] = self.matcher.kb.add_composite_node(
                knowledge, "RelatedTo", ids, mask
            )
        for question_id, question_allowed_facts in zip(
            question_ids, list_of_question_allowed_facts
        ):
            self.allowed_composite_nodes[question_id] = [
                self.fact_to_composite_node[f] for f in question_allowed_facts[:3000]
            ]
        for fact, node in self.fact_to_composite_node.items():
            self.composite_node_to_fact[node] = fact
        logging.info(f"Added {len(tokens)} composite nodes")

    def __reduce__(self):
        return QASCMatcher, (self.tokenizer,)
