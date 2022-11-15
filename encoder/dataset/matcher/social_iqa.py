import re
import os
import logging
from tqdm import tqdm
from typing import List, Dict
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNet, SocialIQA, ATOMIC2020
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.file import PickleCache, JSONCache
from encoder.utils.settings import preprocess_cache_dir, dataset_cache_dir


class SocialIQAMatcher(BaseMatcher):
    ATOMIC_RELATION_TRANSLATION = {
        "AtLocation": "located at",
        "CapableOf": "is capable of",
        "Causes": "causes",
        "CausesDesire": "makes someone want",
        "CreatedBy": "is created by",
        "Desires": "desires",
        "HasA": "has",
        "HasFirstSubevent": "begins with the event",
        "HasLastSubevent": "ends with the event",
        "HasPrerequisite": "to do this requires",
        "HasProperty": "has property",
        "HasSubEvent": "includes the event",
        "HinderedBy": "can be hindered by",
        "InstanceOf": "is an example of",
        "isAfter": "happens after",
        "isBefore": "happens before",
        "isFilledBy": "can be completed by",
        "MadeOf": "is made of",
        "MadeUpOf": "made of",
        "MotivatedByGoal": "is a step towards accomplishing the goal",
        "NotDesires": "does not desire",
        "ObjectUse": "used for",
        "UsedFor": "used for",
        "oEffect": "as a result Y or others will",
        "oReact": "as a result Y or others feels",
        "oWant": "as a result Y or others want",
        "PartOf": "is a part of",
        "ReceivesAction": "can receive action",
        "xAttr": "X is seen as",
        "xEffect": "as a result PersonX will",
        "xIntent": "because PersonX wanted",
        "xNeed": "but before PersonX needed",
        "xReact": "as a result PersonX feels",
        "xReason": "because",
        "xWant": "as a result PersonX wants",
    }

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
        self.social_iqa = SocialIQA().require()
        self.atomic = ATOMIC2020().require()

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

        super(SocialIQAMatcher, self).__init__(tokenizer, matcher, archive_path)
        self.matcher.kb.disable_edges_with_weight_below(1)
        # self.matcher.kb.disable_edges_of_relationships(["HasContext", "RelatedTo"])
        self.matcher.kb.disable_edges_of_relationships(["HasContext"])

        self.allowed_composite_nodes = {}
        self.fact_to_composite_node = {}
        self.composite_node_to_fact = {}

    def add_atomic_knowledge(
        self,
        question_allowed_knowledge: Dict[str, List[str]],
        selected_knowledge: List[str],
        mask: List[str],
    ):
        logging.info("Adding ATOMIC2020 knowledge")

        def generate():
            tokens = [
                (knowledge, token_ids, token_mask)
                for knowledge, (token_ids, token_mask) in zip(
                    selected_knowledge,
                    self.parallel_tokenize_and_mask(selected_knowledge, mask),
                )
            ]
            return len(selected_knowledge), tokens

        with PickleCache(
            os.path.join(
                preprocess_cache_dir, "social_iqa_matcher_atomic_knowledge.data"
            ),
            generate,
        ) as cache:
            hash_value, tokens = cache.data
            for knowledge, token_ids, token_mask in tqdm(tokens):
                self.fact_to_composite_node[
                    knowledge
                ] = self.matcher.kb.add_composite_node(
                    knowledge, "RelatedTo", token_ids, token_mask
                )

            for (
                question_id,
                question_allowed_knowledge,
            ) in question_allowed_knowledge.items():
                self.allowed_composite_nodes[question_id] = [
                    self.fact_to_composite_node[f] for f in question_allowed_knowledge
                ]
            for fact, node in self.fact_to_composite_node.items():
                self.composite_node_to_fact[node] = fact
            logging.info(f"Added {len(tokens)} composite nodes")

    def get_atomic_knowledge_text_and_mask(self):
        def generate():
            raw_knowledge = []
            added = set()
            duplicate_num = 0
            invalid_num = 0
            for path in (
                self.atomic.train_path,
                self.atomic.validate_path,
                self.atomic.test_path,
            ):
                with open(path, "r") as file:
                    for line in file:
                        if line in added:
                            # There are duplicate lines
                            duplicate_num += 1
                            continue
                        added.add(line)
                        segments = line.strip("\n").split("\t")
                        if len(segments) != 3 or segments[-1] == "none":
                            invalid_num += 1
                            continue

                        raw_knowledge.append(segments)

            # Check whether there exists a same edge in ConceptNet
            # some ConceptNet relations are converted in ATOMIC,
            # ignore them
            knowledge = []
            mask = []

            def convert(relation):
                if relation == "xEffect":
                    return "Causes"
                elif relation == "xWant":
                    return "CausesDesire"
                elif relation == "MadeUpOf":
                    return "MadeOf"
                elif relation == "xNeed":
                    return "HasPrerequisite"
                elif relation == "HasSubEvent":
                    return "HasSubevent"
                elif relation == "MadeUpOf":
                    return "PartOf"
                elif relation == "ObjectUse":
                    return "UsedFor"
                else:
                    return relation

            def mask_atomic_stopwords(entity):
                mask = ["+"] * len(entity)
                for mask_part in ("PersonX", "PersonY", "Person X", "Person Y", "___"):
                    start = 0
                    while True:
                        idx = entity.find(mask_part, start)
                        if idx == -1:
                            break
                        start = idx + len(mask_part)
                        for i in range(idx, idx + len(mask_part)):
                            mask[i] = "-"
                return "".join(mask)

            edges = self.matcher.kb.find_edges(
                source_nodes=[k[0] for k in raw_knowledge],
                relations=[convert(k[1]) for k in raw_knowledge],
                target_nodes=[k[2] for k in raw_knowledge],
                quiet=True,
            )
            for k, e in zip(raw_knowledge, edges):
                if e[0] == -1:
                    knowledge.append(
                        "{} {} {}".format(
                            k[0].replace("___", "Z"),
                            self.ATOMIC_RELATION_TRANSLATION[k[1]],
                            k[2],
                        ).lower()
                    )
                    mask.append(
                        "{}-{}-{}".format(
                            mask_atomic_stopwords(k[0].replace("___", "Z")),
                            "-" * len(self.ATOMIC_RELATION_TRANSLATION[k[1]]),
                            mask_atomic_stopwords(k[2]),
                        )
                    )
            print(
                f"Invalid lines: {invalid_num}, duplicate lines: {duplicate_num} "
                f"conceptnet covered lines: {len(raw_knowledge) - len(knowledge)}, "
                f"valid lines: {len(knowledge)}"
            )
            return knowledge, mask

        with JSONCache(
            os.path.join(
                preprocess_cache_dir, "social_iqa_matcher_filtered_corpus.json"
            ),
            generate,
        ) as cache:
            return cache.data

    def __reduce__(self):
        return (
            SocialIQAMatcher,
            (self.tokenizer,),
        )
