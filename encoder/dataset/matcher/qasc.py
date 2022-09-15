import re
import os
import json
import nltk
import pickle
import logging
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNetWithGloVe, OpenBookQA, QASC
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.file import RawTextCache, JSONCache, PickleCache
from encoder.utils.settings import preprocess_cache_dir
from .fact_filter import FactFilter
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

        self.composite_node_index = {}
        self.reverse_composite_node_index = {}
        self.allowed_composite_nodes = {}

        self.added_qasc_corpus_facts = None
        self.add_qasc_corpus()
        self.validate_qasc_corpus_retrieval_rate()

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
            id_to_facts = {}
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
                        id_to_facts[entry["id"]] = [fact1, fact2]
            knowledge = sorted(list(added))
            result = [
                (fact, tokens, mask)
                for fact, (tokens, mask) in zip(
                    knowledge, self.parallel_tokenize_and_mask(knowledge)
                )
            ]
            return id_to_facts, result

        count = 0
        with PickleCache(
            os.path.join(
                preprocess_cache_dir,
                f"qasc_matcher_{'t_v' if train_and_validate else 't'}_facts_knowledge.data",
            ),
            generate,
        ) as cache:
            id_to_facts, result = cache.data
            for knowledge, ids, mask in tqdm(result):
                if knowledge not in self.added_qasc_corpus_facts:
                    count += 1
                    idx = self.composite_node_index[
                        knowledge
                    ] = self.matcher.kb.add_composite_node(
                        knowledge, "RelatedTo", ids, mask
                    )
                    self.reverse_composite_node_index[idx] = knowledge
            for id, facts in id_to_facts.items():
                self.allowed_composite_nodes[id] += [
                    self.composite_node_index[f] for f in facts
                ]
        logging.info(f"Added {count} composite nodes")

    def add_qasc_corpus(self):
        logging.info("Adding QASC corpus")

        if not os.path.exists(
            os.path.join(
                preprocess_cache_dir, "qasc_matcher_allowed_composite_nodes.data"
            )
        ) or not os.path.exists(
            os.path.join(preprocess_cache_dir, "qasc_matcher_selected_corpus.data")
        ):

            ids = []
            allowed_facts = []
            questions = []
            list_of_choices = []
            for dataset_path in (
                self.qasc.train_path,
                self.qasc.validate_path,
                self.qasc.test_path,
            ):
                with open(dataset_path, "r") as file:
                    for line in file:
                        entry = json.loads(line)
                        questions.append(entry["question"]["stem"])
                        ids.append(entry["id"])
                        list_of_choices.append(
                            [choice["text"] for choice in entry["question"]["choices"]]
                        )

            def generate_filtered_facts():
                with open(self.qasc.corpus_path, "r") as file:
                    raw_facts = [line.strip("\n") for line in file]
                fact_filter = FactFilter()
                return fact_filter.clean(raw_facts, remove_incomplete_sentences=False)

            with RawTextCache(
                os.path.join(preprocess_cache_dir, "qasc_matcher_filtered_corpus.json"),
                generate_filtered_facts,
            ) as cache:
                logging.info(f"{len(cache.data)} facts in filtered corpus")
                filtered_facts = cache.data

            def generate_first_level_facts():
                fact_selector = FactSelector(questions, filtered_facts, min_score=0.55)
                return {
                    "queries": questions,
                    "query_facts": fact_selector.selected_facts,
                }

            with JSONCache(
                os.path.join(
                    preprocess_cache_dir, "qasc_matcher_first_level_facts.json"
                ),
                generate_first_level_facts,
            ) as cache:
                for facts in cache.data["query_facts"]:
                    allowed_facts.append(
                        [
                            f.strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                            for f in facts[:50]
                        ]
                    )
                first_level_top_facts = [
                    facts[:10] for facts in cache.data["query_facts"]
                ]

            def generate_first_level_keywords():
                first_level_keywords = []
                lemma = nltk.wordnet.WordNetLemmatizer()
                for question, facts, choices in tqdm(
                    zip(questions, first_level_top_facts, list_of_choices),
                    total=len(questions),
                ):
                    (
                        question_keywords,
                        list_of_fact_keywords,
                        list_of_choice_keywords,
                    ) = (
                        [],
                        [[] for _ in range(len(facts))],
                        [[] for _ in range(len(choices))],
                    )

                    for sentence, keywords in zip(
                        (question, *facts, *choices),
                        (
                            question_keywords,
                            *list_of_fact_keywords,
                            *list_of_choice_keywords,
                        ),
                    ):
                        added = set()
                        tokens = nltk.word_tokenize(sentence)
                        for token, pos in BaseMatcher.safe_pos_tag(tokens):
                            if (
                                pos.startswith("NN")
                                or pos.startswith("JJ")
                                or (
                                    pos.startswith("VB")
                                    and token.lower() not in BaseMatcher.VERB_FILTER_SET
                                )
                            ):
                                lem_token = lemma.lemmatize(token.lower())
                                if lem_token not in self.STOPWORDS_SET:
                                    if lem_token not in added:
                                        added.add(lem_token)
                                        keywords.append(lem_token)
                    first_level_keywords.append(
                        {
                            "question": question_keywords,
                            "facts": list_of_fact_keywords,
                            "choices": list_of_choice_keywords,
                        }
                    )
                return first_level_keywords

            with JSONCache(
                os.path.join(
                    preprocess_cache_dir, "qasc_matcher_first_level_keywords.json"
                ),
                generate_first_level_keywords,
            ) as cache:
                first_level_keywords = cache.data

            def generate_second_level_facts():
                queries = []
                query_indices = []
                for idx, keywords in enumerate(first_level_keywords):
                    sub_queries = []
                    for fact_keywords in keywords["facts"]:
                        for choice_keywords in keywords["choices"]:
                            order = (
                                keywords["question"] + choice_keywords + fact_keywords
                            )
                            first_set = set(keywords["question"] + choice_keywords)
                            second_set = set(fact_keywords)
                            unordered_query_keywords = first_set.union(
                                second_set
                            ).difference(first_set.intersection(second_set))
                            unordered_query_keywords = [
                                (k, order.index(k)) for k in unordered_query_keywords
                            ]
                            ordered_query_keywords = [
                                k[0]
                                for k in sorted(
                                    unordered_query_keywords, key=lambda k: k[1]
                                )
                            ]
                            base_query = " ".join(ordered_query_keywords)
                            sub_queries.append(base_query)
                            query_indices.append(idx)
                    queries.append(sub_queries)

                fact_selector = FactSelector(
                    [q for sub_q in queries for q in sub_q],
                    filtered_facts,
                    min_score=0.55,
                    inner_batch_size=2048,
                )
                selected_facts = [[] for _ in range(len(first_level_keywords))]
                for idx, list_of_facts in zip(
                    query_indices, fact_selector.selected_facts
                ):
                    selected_facts[idx].append(list_of_facts)
                return {
                    "list_of_queries": queries,
                    "list_of_query_facts": selected_facts,
                }

            with JSONCache(
                os.path.join(
                    preprocess_cache_dir, "qasc_matcher_second_level_facts.json"
                ),
                generate_second_level_facts,
            ) as cache:
                for idx, list_of_facts in enumerate(cache.data["list_of_query_facts"]):
                    for facts in list_of_facts:
                        allowed_facts[idx] += [
                            f.strip("\n")
                            .strip(".")
                            .strip('"')
                            .strip("'")
                            .strip(",")
                            .lower()
                            for f in facts[:40]
                        ]
            facts = set(f for facts in allowed_facts for f in facts)
            self.added_qasc_corpus_facts = facts
            facts = sorted(list(facts))

            def generate_selected_qasc_corpus_tokens():
                result = [
                    (fact, tokens, mask)
                    for fact, (tokens, mask) in zip(
                        facts, self.parallel_tokenize_and_mask(facts)
                    )
                ]
                return result

            with PickleCache(
                os.path.join(preprocess_cache_dir, "qasc_matcher_selected_corpus.data"),
                generate_selected_qasc_corpus_tokens,
            ) as cache:
                for knowledge, tokens, mask in tqdm(cache.data):
                    idx = self.composite_node_index[
                        knowledge
                    ] = self.matcher.kb.add_composite_node(
                        knowledge, "RelatedTo", tokens, mask
                    )
                    self.reverse_composite_node_index[idx] = knowledge
                logging.info(f"Added {len(cache.data)} composite nodes")

            def generate_allowed_composite_nodes():
                logging.info("Indexing...")
                allowed_composite_nodes = {}
                for id, facts in tqdm(zip(ids, allowed_facts), total=len(ids)):
                    allowed_composite_nodes[id] = [
                        self.composite_node_index[f] for f in facts
                    ]
                return allowed_composite_nodes

            with PickleCache(
                os.path.join(
                    preprocess_cache_dir, "qasc_matcher_allowed_composite_nodes.data"
                ),
                generate_allowed_composite_nodes,
            ) as cache:
                self.allowed_composite_nodes = cache.data
        else:
            self.added_qasc_corpus_facts = set()
            with open(
                os.path.join(preprocess_cache_dir, "qasc_matcher_selected_corpus.data"),
                "rb",
            ) as file:
                data = pickle.load(file)
                for knowledge, tokens, mask in tqdm(data):
                    idx = self.composite_node_index[
                        knowledge
                    ] = self.matcher.kb.add_composite_node(
                        knowledge, "RelatedTo", tokens, mask
                    )
                    self.reverse_composite_node_index[idx] = knowledge
                    self.added_qasc_corpus_facts.add(knowledge)
                logging.info(f"Added {len(data)} composite nodes")

            with open(
                os.path.join(
                    preprocess_cache_dir, "qasc_matcher_allowed_composite_nodes.data"
                ),
                "rb",
            ) as file:
                self.allowed_composite_nodes = pickle.load(file)

    def validate_qasc_corpus_retrieval_rate(self):
        for split, path in zip(
            ("train", "validate"), (self.qasc.train_path, self.qasc.validate_path)
        ):
            fact1_retrieved_count = 0
            fact2_retrieved_count = 0
            total = 0
            with open(path, "r") as file:
                for line in file:
                    total += 1
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
                    if (
                        fact1 in self.composite_node_index
                        and self.composite_node_index[fact1]
                        in self.allowed_composite_nodes[entry["id"]]
                    ):
                        fact1_retrieved_count += 1
                    if (
                        fact2 in self.composite_node_index
                        and self.composite_node_index[fact2]
                        in self.allowed_composite_nodes[entry["id"]]
                    ):
                        fact2_retrieved_count += 1
            logging.info(
                f"{split}-Fact1 retrieved ratio: {fact1_retrieved_count / total}"
            )
            logging.info(
                f"{split}-Fact2 retrieved ratio: {fact2_retrieved_count / total}"
            )
            logging.info(
                f"{split}-retrieved length: {(fact1_retrieved_count + fact2_retrieved_count) / total}"
            )

    def __reduce__(self):
        return QASCMatcher, (self.tokenizer,)
