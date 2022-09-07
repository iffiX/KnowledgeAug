import re
import os
import json
import nltk
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
        self.matcher.kb.disable_edges_of_relationships(["RelatedTo"])

        # Add knowledge from openbook QA as composite nodes
        # self.add_openbook_qa_knowledge()
        self.added_qasc_corpus_facts = None
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
            for dataset_path in paths:
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        added.add(
                            entry["fact1"]
                            .strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                        )
                        added.add(
                            entry["fact2"]
                            .strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                        )
            knowledge = list(added)
            result = [
                (fact, ids, mask)
                for fact, (ids, mask) in zip(
                    knowledge, self.parallel_tokenize_and_mask(knowledge)
                )
            ]
            return result

        count = 0
        with PickleCache(
            os.path.join(
                preprocess_cache_dir,
                f"qasc_{'train_val' if train_and_validate else 'train'}_facts_knowledge.data",
            ),
            generate,
        ) as cache:
            for knowledge, ids, mask in tqdm(cache.data):
                if knowledge not in self.added_qasc_corpus_facts:
                    count += 1
                    self.matcher.kb.add_composite_node(
                        knowledge, "RelatedTo", ids, mask
                    )
        logging.info(f"Added {count} composite nodes")

    def add_openbook_qa_knowledge(self):
        logging.info("Adding OpenBook QA knowledge")

        def generate():
            knowledge = []
            for path in (
                self.openbook_qa.facts_path,
                self.openbook_qa.crowd_source_facts_path,
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
                        if len(line) < 3:
                            continue
                        knowledge.append(line)
            result = [
                (fact, ids, mask)
                for fact, (ids, mask) in zip(
                    knowledge, self.parallel_tokenize_and_mask(knowledge)
                )
            ]
            return result

        with PickleCache(
            os.path.join(preprocess_cache_dir, "qasc_obqa_knowledge.data"), generate,
        ) as cache:
            for knowledge, ids, mask in tqdm(cache.data):
                self.matcher.kb.add_composite_node(knowledge, "RelatedTo", ids, mask)
        logging.info(f"Added {len(cache.data)} composite nodes")

    def add_qasc_corpus(self):
        logging.info("Adding QASC corpus")

        def generate_first_level_facts():
            questions = []
            for dataset_path in (
                self.qasc.train_path,
                self.qasc.validate_path,
                self.qasc.test_path,
            ):
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        questions.append(entry["question"]["stem"])
            with open(self.qasc.corpus_path, "r") as file:
                raw_facts = [line.strip("\n") for line in file]
            fact_selector = FactSelector(questions, raw_facts, min_score=0.55)
            return {
                "queries": questions,
                "query_facts": fact_selector.selected_facts,
                "query_facts_rank": fact_selector.selected_facts_rank,
            }

        with JSONCache(
            os.path.join(preprocess_cache_dir, "qasc_first_level_facts.json"),
            generate_first_level_facts,
        ) as cache:
            first_level_facts = list(
                set(
                    [
                        f.strip("\n")
                        .strip(".")
                        .strip('"')
                        .strip("'")
                        .strip(",")
                        .lower()
                        for facts in cache.data["query_facts"]
                        for f in facts[:10]
                    ]
                )
            )
            first_level_top_facts = [facts[:5] for facts in cache.data["query_facts"]]

        def generate_first_level_keywords():
            first_level_keywords = []
            lemma = nltk.wordnet.WordNetLemmatizer()
            for facts in tqdm(first_level_top_facts):
                tokens = nltk.word_tokenize(" ".join(facts))
                keywords = set()
                for token, pos in BaseMatcher.safe_pos_tag(tokens):
                    if pos.startswith("NN") or pos.startswith("JJ"):
                        lem_token = lemma.lemmatize(token.lower())
                        if lem_token not in self.STOPWORDS_SET:
                            keywords.add(lem_token)
                first_level_keywords.append(list(keywords))
            return first_level_keywords

        with JSONCache(
            os.path.join(preprocess_cache_dir, "qasc_first_level_keywords.json"),
            generate_first_level_keywords,
        ) as cache:
            first_level_keywords = cache.data

        def generate_second_level_facts():
            list_of_choices = []
            queries = []
            for dataset_path in (
                self.qasc.train_path,
                self.qasc.validate_path,
                self.qasc.test_path,
            ):
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        list_of_choices.append(
                            [choice["text"] for choice in entry["question"]["choices"]]
                        )
            for keywords, choices in zip(first_level_keywords, list_of_choices):
                base_query = " ".join(keywords)
                for choice in choices:
                    queries.append(base_query + " " + choice)
            with open(self.qasc.corpus_path, "r") as file:
                raw_facts = [line.strip("\n") for line in file]
            fact_selector = FactSelector(queries, raw_facts, min_score=0.55)
            return {
                "queries": queries,
                "query_facts": fact_selector.selected_facts,
                "query_facts_rank": fact_selector.selected_facts_rank,
            }

        with JSONCache(
            os.path.join(preprocess_cache_dir, "qasc_second_level_facts.json"),
            generate_second_level_facts,
        ) as cache:
            second_level_facts = list(
                set(
                    [
                        f.strip("\n")
                        .strip(".")
                        .strip('"')
                        .strip("'")
                        .strip(",")
                        .lower()
                        for facts in cache.data["query_facts"]
                        for f in facts[:5]
                    ]
                )
            )
        facts = list(set(first_level_facts + second_level_facts))
        self.added_qasc_corpus_facts = set(facts)

        def generate_filtered_qasc_corpus_tokens():
            result = [
                (fact, ids, mask)
                for fact, (ids, mask) in zip(
                    facts, self.parallel_tokenize_and_mask(facts)
                )
            ]
            return result

        with PickleCache(
            os.path.join(preprocess_cache_dir, "qasc_filtered_corpus.data"),
            generate_filtered_qasc_corpus_tokens,
        ) as cache:
            for knowledge, ids, mask in tqdm(cache.data):
                self.matcher.kb.add_composite_node(knowledge, "RelatedTo", ids, mask)
            logging.info(f"Added {len(cache.data)} composite nodes")

    def __reduce__(self):
        return QASCMatcher, (self.tokenizer,)
