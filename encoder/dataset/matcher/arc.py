import re
import os
import json
import tqdm
import pickle
import logging
from nltk import tokenize
from transformers import PreTrainedTokenizerBase
from .fact_filter import FactFilter
from encoder.dataset.download import (
    ConceptNetWithGloVe,
    OpenBookQA,
    ARC,
    QASC,
    UnifiedQAIR,
)
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.file import JSONCache, PickleCache
from encoder.utils.settings import preprocess_cache_dir


class ARCMatcher(BaseMatcher):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive-glove.data")
        )
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding-glove.hdf5")
        )
        self.concept_net = ConceptNetWithGloVe().require()
        self.openbook_qa = OpenBookQA().require()
        self.arc = ARC().require()
        self.qasc = QASC().require()
        self.unifiedqa_ir = UnifiedQAIR().require()

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
        super(ARCMatcher, self).__init__(tokenizer, matcher, archive_path)

        matcher.kb.enable_edges_of_relationships(
            ["DefinedAs", "DerivedFrom", "FormOf", "Synonym", "Antonym", "IsA", "HasA"]
        )
        self.matcher.kb.disable_edges_with_weight_below(1)

        # Add knowledge as composite nodes
        self.add_additional_knowledge()

    def add_additional_knowledge(self):
        with JSONCache(
            os.path.join(preprocess_cache_dir, "arc_facts.json"), self.generate_facts
        ) as facts_cache:
            facts = facts_cache.data

        def tokenize():
            data = []
            logging.info("Tokenizing knowledge")
            for fact in tqdm.tqdm(facts):
                fact = fact.strip("\n").strip(".").strip('"').strip("'").strip(",")
                if len(fact) < 3:
                    continue
                ids, mask = self.tokenize_and_mask(fact)
                data.append((fact, ids, mask))
            return data

        with PickleCache(
            os.path.join(preprocess_cache_dir, "arc_facts_tokenized.data"), tokenize
        ) as token_cache:
            logging.info("Adding additional knowledge")
            for fact, ids, mask in tqdm.tqdm(token_cache.data):
                self.matcher.kb.add_composite_node(fact, "RelatedTo", ids, mask)
            logging.info(f"Added {len(token_cache.data)} composite nodes")

    def generate_facts(self):
        facts = set()
        logging.info(f"Generating facts")
        for path in (
            self.qasc.train_path,
            self.qasc.validate_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    entry = json.loads(line)
                    facts.add(entry["fact1"].strip(".").lower())
                    facts.add(entry["fact2"].strip(".").lower())
        for path in (
            self.openbook_qa.facts_path,
            self.openbook_qa.crowd_source_facts_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    line = line.replace("\n", "").replace('"', "").replace(".", "")
                    if len(line) < 3:
                        continue
                    facts.add(line.lower())

        entries = []
        unifiedqa_ir_facts = []
        for path in (
            self.unifiedqa_ir.train_path,
            self.unifiedqa_ir.validate_path,
            self.unifiedqa_ir.test_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    entries.append(json.loads(line))
        for entry in tqdm.tqdm(entries):
            for sentence in tokenize.sent_tokenize(
                re.sub(r"(\. ){2,}", "...", entry["para"])
            ):
                unifiedqa_ir_facts.append(sentence.lower())
        unifiedqa_ir_facts = FactFilter().clean(unifiedqa_ir_facts)

        # save for checking
        path = os.path.join(preprocess_cache_dir, "arc_unified_qa_facts_debug.json")
        with open(path, "w") as file:
            json.dump(unifiedqa_ir_facts, file, indent=2)

        for fact in unifiedqa_ir_facts:
            facts.add(fact)
        logging.info(f"Generated {len(facts)} facts")
        return sorted(list(facts))

    def __reduce__(self):
        return ARCMatcher, (self.tokenizer,)
