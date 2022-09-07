import re
import tqdm
import nltk
import spacy
import logging
import multiprocessing as mp
from typing import List, Dict, Tuple, Union
from nltk.corpus import stopwords
from transformers import PreTrainedTokenizerBase
from encoder.dataset.matcher import KnowledgeMatcher

Edge = Tuple[int, int, int, float, str]
Tokens = List[int]


class BaseMatcher:
    TOKENIZER_INSTANCE = None
    VERB_FILTER_SET = {
        "do",
        "did",
        "does",
        "done",
        "have",
        "having",
        "has",
        "had",
        "be",
        "am",
        "is",
        "are",
        "being",
        "was",
        "were",
    }
    NATURAL_TEMPLATES = {
        "Antonym": "{0} is an antonym of {1}",
        "AtLocation": "{0} is located at {1}",
        "CapableOf": "{0} can {1}",
        "Causes": "{0} causes {1}",
        "CausesDesire": "{0} would make you want to {1}",
        "CreatedBy": "{0} is created by {1}",
        "DefinedAs": "{0} is defined as {1}",
        "DerivedFrom": "{0} is derived from {1}",
        "Desires": "{0} wants {1}",
        "DistinctFrom": "{0} is different from {1}",
        "Entails": "{0} entails {1}",
        "EtymologicallyDerivedFrom": "etymologically {0} is derived from {1}",
        "EtymologicallyRelatedTo": "etymologically {0} is related to {1}",
        "FormOf": "{0} is a word form of {1}",
        "HasA": "{0} has {1}",
        "HasContext": "the context of {0} is {1}",
        "HasFirstSubevent": "the first thing you do when you {0} is {1}",
        "HasLastSubevent": "the last thing you do when you {0} is {1}",
        "HasPrerequisite": "{0} requires {1}",
        "HasProperty": "{0} has property {1}",
        "HasSubevent": "something you might do while {0} is {1}",
        "InstanceOf": "{0} is an instance of {1}",
        "IsA": "{0} is a {1}",
        "LocatedNear": "{0} is located near {1}",
        "MadeOf": "{0} is made of {1}",
        "MannerOf": "{0} is a way to {1}",
        "MotivatedByGoal": "you would {0} because you want to {1}",
        "NotCapableOf": "{0} do not {1}",
        "NotDesires": "{0} doesn't want {1}",
        "NotHasProperty": "{0} is not {1}",
        "PartOf": "{0} is a part of {1}",
        "ReceivesAction": "{0} is {1}",
        "RelatedTo": "{0} is related to {1}",
        "SimilarTo": "{0} is similar to {1}",
        "SymbolOf": "{0} is a symbol of {1}",
        "Synonym": "{0} is a synonym of {1}",
        "UsedFor": "{0} is for {1}",
        "capital": "{1} is the capital of {0}",
        "field": "{0} is in the field of {1}",
        "genre": "{0} is in the genre of {1}",
        "genus": "{0} is in a genus of {1}",
        "influencedBy": "{0} is influenced by {1}",
        "knownFor": "{0} is known for {1}",
        "language": "{0} is the language of {1}",
        "leader": "the leader of {0} is {1}",
        "occupation": "the occupation of {0} is {1}",
        "product": "{0} produces {1}",
    }

    STANDARD_TEMPLATES = {
        "Antonym": "({0}) <antonym> ({1})",
        "AtLocation": "({0}) <at location> ({1})",
        "CapableOf": "({0}) <capable of> ({1})",
        "Causes": "({0}) <causes> ({1})",
        "CausesDesire": "({0}) <causes desire> ({1})",
        "CreatedBy": "({0}) <created by> ({1})",
        "DefinedAs": "({0}) <defined as> ({1})",
        "DerivedFrom": "({0}) <derived from> ({1})",
        "Desires": "({0}) <desires> ({1})",
        "DistinctFrom": "({0}) <distinct from> ({1})",
        "Entails": "({0}) <entails> ({1})",
        "EtymologicallyDerivedFrom": "({0}) <etymologically derived from> ({1})",
        "EtymologicallyRelatedTo": "({0}) <etymologically related to> ({1})",
        "FormOf": "({0}) <etymologically form of> ({1})",
        "HasA": "({0}) <has a> ({1})",
        "HasContext": "({0}) <has context> ({1})",
        "HasFirstSubevent": "({0}) <has first sub event> ({1})",
        "HasLastSubevent": "({0}) <has last sub event> ({1})",
        "HasPrerequisite": "({0}) <has prerequisite> ({1})",
        "HasProperty": "({0}) <has property> ({1})",
        "HasSubevent": "({0}) <has sub event> ({1})",
        "InstanceOf": "({0}) <instance of> ({1})",
        "IsA": "({0}) <is a> ({1})",
        "LocatedNear": "({0}) <located near> ({1})",
        "MadeOf": "({0}) <made of> ({1})",
        "MannerOf": "({0}) <manner of> ({1})",
        "MotivatedByGoal": "({0}) <motivated by goal> ({1})",
        "NotCapableOf": "({0}) <not capable of> ({1})",
        "NotDesires": "({0}) <not desires> ({1})",
        "NotHasProperty": "({0}) <not has property> ({1})",
        "PartOf": "({0}) <part of> ({1})",
        "ReceivesAction": "({0}) <receives action> ({1})",
        "RelatedTo": "({0}) <related to> ({1})",
        "SimilarTo": "({0}) <similar to> ({1})",
        "SymbolOf": "({0}) <symbol of> ({1})",
        "Synonym": "({0}) <synonym> ({1})",
        "UsedFor": "({0}) <used for> ({1})",
        "capital": "({0}) <capital> ({1})",
        "field": "({0}) <field> ({1})",
        "genre": "({0}) <genre> ({1})",
        "genus": "({0}) <genus> ({1})",
        "influencedBy": "({0}) <influenced by> ({1})",
        "knownFor": "({0}) <known for> ({1})",
        "language": "({0}) <language> ({1})",
        "leader": "({0}) <leader> ({1})",
        "occupation": "({0}) <occupation> ({1})",
        "product": "({0}) <product> ({1})",
    }
    STOPWORDS_SET = set(stopwords.words("english"))

    def __init__(self, tokenizer: PreTrainedTokenizerBase, matcher: KnowledgeMatcher):
        self.tokenizer = tokenizer
        self.matcher = matcher
        self.nlp = None
        self.natural_relationship_templates = [
            self.NATURAL_TEMPLATES[rel] for rel in matcher.kb.relationships
        ]
        self.standard_relationship_templates = [
            self.STANDARD_TEMPLATES[rel] for rel in matcher.kb.relationships
        ]
        self.standard_no_symbol_templates = [
            re.sub("\(|\)|<|>", "", self.STANDARD_TEMPLATES[rel])
            for rel in matcher.kb.relationships
        ]

    def find_closest_concept(self, target_concept: str, concepts: List[str]):
        if self.nlp is None:
            self.nlp = spacy.load("en_core_web_md")
        try:
            return self.matcher.find_closest_concept(target_concept, concepts)
        except ValueError:
            target_doc = self.nlp(target_concept)
            similarity = [
                target_doc.similarity(self.nlp(concept)) for concept in concepts
            ]
            closest = concepts[similarity.index(max(similarity))]
            logging.info(
                f"Concept {target_concept} not found, "
                f"use spacy to match, closest {closest}"
            )
            return closest

    def find_shortest_path(
        self,
        source_sentence: str,
        target_sentence: str,
        intermediate_nodes: List[str],
        source_mask: str = "",
        target_mask: str = "",
        max_depth_for_each_node: int = 3,
        split_node_minimum_edge_num: int = 20,
        split_node_minimum_similarity: float = 0.35,
    ) -> Tuple[List[List[str]], List[List[Edge]], List[List[int]]]:
        """
        Returns:
            A list containing several sub list of edge annotations,
            each sub list corresponding to a sub path.
            A list of sub list of corresponding edges.
            a list of starting nodes of each sub path, note it has one more level
            than other two lists which contains end nodes of last path.
        """
        source_tokens, _source_mask = self.tokenize_and_mask(
            source_sentence, source_mask, mask_stopwords=True
        )
        target_tokens, _target_mask = self.tokenize_and_mask(
            target_sentence, target_mask, mask_stopwords=True
        )
        (
            list_of_sub_path_annotation_tokens,
            list_of_sub_path_edges,
            starting_nodes,
        ) = self.matcher.find_shortest_path(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            intermediate_nodes=intermediate_nodes,
            source_mask=_source_mask,
            target_mask=_target_mask,
            max_depth_for_each_node=max_depth_for_each_node,
            split_node_minimum_edge_num=split_node_minimum_edge_num,
            split_node_minimum_similarity=split_node_minimum_similarity,
        )
        result = (
            [
                [self.tokenizer.decode(tokens) for tokens in sub_path_tokens]
                for sub_path_tokens in list_of_sub_path_annotation_tokens
            ],
            # self.sub_paths_to_annotations(list_of_sub_path_edges, "standard_no_symbol"),
            list_of_sub_path_edges,
            starting_nodes,
        )
        return result

    def find_available_choices(
        self,
        visited_nodes: List[int],
        start_nodes: List[int],
        target_nodes: List[int],
        max_depth: int = 2,
        only_target: bool = False,
        filter_composite_nodes_by_f_beta: bool = False,
        minimum_f_beta: float = 0,
    ) -> Tuple[List[List[str]], List[List[int]], List[int], List[List[Edge]]]:
        """
        Returns:
            A list containing several sub list of edge annotations,
            each sub list corresponding to a sub path.
            A list of next node ids the path is leading to. If the path ends with 
            a composite node, its component nodes are returned.
             A list of next node ids the path is leading to.
            A list of sub list of corresponding edges.
        """
        (
            list_of_sub_path_annotation_tokens,
            list_of_sub_path_next_nodes,
            list_of_sub_path_raw_next_nodes,
            list_of_sub_path_edges,
        ) = self.matcher.find_available_choices(
            visited_nodes,
            start_nodes,
            target_nodes,
            max_depth,
            only_target,
            filter_composite_nodes_by_f_beta,
            minimum_f_beta,
        )
        return (
            [
                self.tokenizer.batch_decode(sub_path_tokens)
                for sub_path_tokens in list_of_sub_path_annotation_tokens
            ],
            # self.sub_paths_to_annotations(list_of_sub_path_edges, "standard_no_symbol"),
            list_of_sub_path_next_nodes,
            list_of_sub_path_raw_next_nodes,
            list_of_sub_path_edges,
        )

    def sub_paths_to_annotations(
        self,
        sub_paths: List[List[Edge]],
        templates: Union[str, List[str]] = "natural",
        prioritize_original_annotation: bool = True,
        lower_case: bool = True,
    ) -> List[List[str]]:
        if isinstance(templates, str):
            if templates == "natural":
                templates = self.natural_relationship_templates
            elif templates == "standard":
                templates = self.standard_relationship_templates
                prioritize_original_annotation = False
            elif templates == "standard_no_symbol":
                templates = self.standard_no_symbol_templates
                prioritize_original_annotation = False
            else:
                raise ValueError(f"Unknown templates configuration: {templates}")
        return self.matcher.sub_paths_to_annotations(
            sub_paths,
            templates,
            prioritize_original_annotation=prioritize_original_annotation,
            lower_case=lower_case,
        )

    def match_source_and_target_nodes(
        self,
        source_sentence: str,
        target_sentence: str,
        source_mask: str = "",
        target_mask: str = "",
        split_node_minimum_edge_num: int = 20,
        split_node_minimum_similarity: float = 0.35,
    ) -> Tuple[List[int], List[int]]:
        """
        Returns:
            A list of node ids in the source sentence,
            and a list of node ids in the target sentence.
        """
        source_tokens, _source_mask = self.tokenize_and_mask(
            source_sentence, source_mask
        )
        target_tokens, _target_mask = self.tokenize_and_mask(
            target_sentence, target_mask
        )
        return self.matcher.match_source_and_target_nodes(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=_source_mask,
            target_mask=_target_mask,
            split_node_minimum_edge_num=split_node_minimum_edge_num,
            split_node_minimum_similarity=split_node_minimum_similarity,
        )

    def match_by_node_embedding(
        self,
        source_sentence: str,
        target_sentence: str = "",
        source_mask: str = "",
        target_mask: str = "",
        disabled_nodes: List[int] = None,
        max_times: int = 1000,
        max_depth: int = 3,
        seed: int = -1,
        edge_top_k: int = -1,
        source_context_range: int = 0,
        trim_path: bool = True,
        split_node_minimum_edge_num: int = 20,
        split_node_minimum_similarity: float = 0.35,
        stop_searching_edge_if_similarity_below: float = 0,
        source_context_weight: float = 0.5,
    ):
        """
        Returns:
            A match result object that can be unified or be used to select paths
        """
        source_tokens, _source_mask = self.tokenize_and_mask(
            source_sentence, source_mask
        )
        if len(target_sentence) == 0:
            target_tokens, _target_mask = source_tokens, _source_mask
        else:
            target_tokens, _target_mask = self.tokenize_and_mask(
                target_sentence, target_mask
            )

        optional_args = {}
        if disabled_nodes is not None:
            optional_args["disabled_nodes"] = disabled_nodes
        result = self.matcher.match_by_node_embedding(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=_source_mask,
            target_mask=_target_mask,
            max_times=max_times,
            max_depth=max_depth,
            seed=seed,
            edge_top_k=edge_top_k,
            source_context_range=source_context_range,
            trim_path=trim_path,
            split_node_minimum_edge_num=split_node_minimum_edge_num,
            split_node_minimum_similarity=split_node_minimum_similarity,
            stop_searching_edge_if_similarity_below=stop_searching_edge_if_similarity_below,
            source_context_weight=source_context_weight,
            **optional_args,
        )
        return result

    def unify_match(self, matches: list):
        """
        Unify several match result into one.
        """
        return self.matcher.join_match_results(matches)

    def select_paths(
        self,
        match,
        max_edges: int = 10,
        discard_edges_if_rank_below: Union[float, str] = 0,
        filter_short_accurate_paths: bool = False,
    ) -> Dict[int, Tuple[int, List[Tokens], List[float]]]:
        if isinstance(discard_edges_if_rank_below, str):
            if discard_edges_if_rank_below != "auto":
                raise ValueError(
                    "discard_edges_if_rank_below can only be set to float or 'auto'"
                )
            discard_edges_if_rank_below = max(-match.target_node_num / 40 + 0.325, 0.2)
        return self.matcher.select_paths(
            match, max_edges, discard_edges_if_rank_below, filter_short_accurate_paths
        )

    def selection_to_list_of_strings(
        self, selection: Dict[int, Tuple[int, List[Tokens], List[float]]],
    ) -> List[str]:
        """
        Returns List of knowledge sequences
        """
        knowledge_tokens = list(
            v for _, (__, v, ___) in selection.items()
        )  # type: List[List[Tokens]]
        knowledge = []
        for kt_list in knowledge_tokens:
            for kt in kt_list:
                knowledge.append(self.tokenizer.decode(kt))
        return knowledge

    def insert_selection(
        self,
        sentence: str,
        selection: Dict[int, Tuple[int, List[Tokens], List[float]]],
        begin: str = "(",
        sep: str = ",",
        end: str = ")",
        insert_at_end: bool = False,
        include_weights: bool = False,
    ) -> str:
        sentence_tokens, _ = self.tokenize_and_mask(sentence)
        if len(selection) == 0:
            return self.tokenizer.decode(sentence_tokens)
        begin_tokens = self.tokenizer.encode(begin, add_special_tokens=False)
        end_tokens = self.tokenizer.encode(end, add_special_tokens=False)
        sep_tokens = self.tokenizer.encode(sep, add_special_tokens=False)
        new_matches = {}
        for pos, (_, edges, weights) in selection.items():
            new_edges = []
            for i, (edge, weight) in enumerate(zip(edges, weights)):
                if not insert_at_end and i == 0:
                    new_edges += begin_tokens
                if include_weights:
                    new_edges += edge + self.tokenizer.encode(
                        f"{weight:.1f}", add_special_tokens=False
                    )
                else:
                    new_edges += edge
                if not insert_at_end and i == len(edges) - 1:
                    new_edges += end_tokens
                else:
                    new_edges += sep_tokens
            new_matches[pos] = new_edges
        sorted_selection = list(
            (k, v) for k, v in new_matches.items()
        )  # type: List[Tuple[int, Tokens]]
        sorted_selection = sorted(sorted_selection, key=lambda x: x[0])
        if insert_at_end:
            sentence_tokens += begin_tokens
            for ss in sorted_selection:
                sentence_tokens = sentence_tokens + ss[1]
            sentence_tokens += end_tokens
        else:
            offset = 0
            for ss in sorted_selection:
                pos = ss[0] + offset
                sentence_tokens = sentence_tokens[:pos] + ss[1] + sentence_tokens[pos:]
                offset += len(ss[1])
        return self.tokenizer.decode(sentence_tokens)

    def insert_selection_at_end_preserve_case(
        self,
        sentence: str,
        selection: Dict[int, Tuple[int, List[Tokens], List[float]]],
        begin: str = "(",
        sep: str = ",",
        end: str = ")",
        include_weights: bool = False,
    ) -> str:
        if len(selection) == 0:
            return sentence
        begin_tokens = self.tokenizer.encode(begin, add_special_tokens=False)
        end_tokens = self.tokenizer.encode(end, add_special_tokens=False)
        sep_tokens = self.tokenizer.encode(sep, add_special_tokens=False)
        new_matches = {}
        knowledge_tokens = []
        for pos, (_, edges, weights) in selection.items():
            new_edges = []
            for i, (edge, weight) in enumerate(zip(edges, weights)):
                if include_weights:
                    new_edges += edge + self.tokenizer.encode(
                        f"{weight:.1f}", add_special_tokens=False
                    )
                else:
                    new_edges += edge
                new_edges += sep_tokens
            new_matches[pos] = new_edges
        sorted_selection = list(
            (k, v) for k, v in new_matches.items()
        )  # type: List[Tuple[int, Tokens]]
        sorted_selection = sorted(sorted_selection, key=lambda x: x[0])

        knowledge_tokens += begin_tokens
        for ss in sorted_selection:
            knowledge_tokens = knowledge_tokens + ss[1]
        knowledge_tokens += end_tokens
        return sentence + " " + self.tokenizer.decode(knowledge_tokens)

    def parallel_tokenize_and_mask(
        self,
        sentences: List[str],
        sentence_masks: List[str] = None,
        mask_stopwords: bool = False,
    ):
        """
        Parallel version of tokenize_and_mask
        """
        sentence_masks = sentence_masks or [""] * len(sentences)
        mask_stopwords = [mask_stopwords] * len(sentences)
        result = {}
        with mp.Pool(
            initializer=self._parallel_tokenize_and_mask_intialize_pool,
            initargs=(self.tokenizer,),
        ) as pool, tqdm.tqdm(total=len(sentences)) as pbar:
            for sentence, ids_and_mask in pool.imap_unordered(
                self._parallel_tokenize_and_mask,
                zip(sentences, sentence_masks, mask_stopwords),
            ):
                pbar.update()
                result[sentence] = ids_and_mask
        return [result[sentence] for sentence in sentences]

    def tokenize_and_mask(
        self, sentence: str, sentence_mask: str = "", mask_stopwords: bool = False
    ):
        """
        Args:
            sentence: A sentence to be tagged by Part of Speech (POS)
            sentence_mask: A string same length as sentence, comprised of "+" and "-", where
                "+" indicates allowing the word overlapped with that position to be matched
                and "-" indicates the position is disallowed.
            mask_stopwords: Whether masking stopwords or not.
        Returns:
            A list of token ids.
            A list of POS mask.
        """
        return self._tokenize_and_mask(
            self.tokenizer, sentence, sentence_mask, mask_stopwords
        )

    @staticmethod
    def _parallel_tokenize_and_mask_intialize_pool(tokenizer):
        BaseMatcher.TOKENIZER_INSTANCE = tokenizer

    @staticmethod
    def _parallel_tokenize_and_mask(args):
        sentence, sentence_mask, mask_stopwords = args
        return (
            sentence,
            BaseMatcher._tokenize_and_mask(
                BaseMatcher.TOKENIZER_INSTANCE, sentence, sentence_mask, mask_stopwords
            ),
        )

    @staticmethod
    def _tokenize_and_mask(
        tokenizer, sentence: str, sentence_mask: str = "", mask_stopwords: bool = False
    ):
        if len(sentence) == 0:
            return [], []
        use_mask = False
        if len(sentence_mask) != 0:
            mask_characters = set(sentence_mask)
            if not mask_characters.issubset({"+", "-"}):
                raise ValueError(
                    f"Sentence mask should only be comprised of '+' "
                    f"and '-',"
                    f" but got {sentence_mask}"
                )
            elif len(sentence) != len(sentence_mask):
                raise ValueError(
                    f"Sentence mask should be the same length as the " f"sentence."
                )
            use_mask = True

        tokens = nltk.word_tokenize(sentence)

        offset = 0
        masks = []
        ids = []
        allowed_tokens = []
        for token, pos in BaseMatcher.safe_pos_tag(tokens):
            token_position = sentence.find(token, offset)
            offset = token_position + len(token)

            # Relaxed matching, If any part is not masked, allow searching for that part
            ids.append(tokenizer.encode(token, add_special_tokens=False))
            if (
                (not use_mask or "+" in set(sentence_mask[token_position:offset]))
                and (
                    pos.startswith("NN")
                    or pos.startswith("JJ")
                    or pos.startswith("RB")
                    or (
                        pos.startswith("VB")
                        and token.lower() not in BaseMatcher.VERB_FILTER_SET
                    )
                    or pos.startswith("CD")
                )
                and (not mask_stopwords or token not in BaseMatcher.STOPWORDS_SET)
            ):
                allowed_tokens.append(token)
                # noun, adjective, adverb, verb
                masks.append([1] * len(ids[-1]))
            else:
                masks.append([0] * len(ids[-1]))
        logging.debug(
            f"Tokens allowed for matching: {allowed_tokens} from sentence {sentence}"
        )
        return [i for iid in ids for i in iid], [m for mask in masks for m in mask]

    @staticmethod
    def safe_pos_tag(tokens):
        # In case the input sentence is a decoded sentence with special characters
        # used in tokenizers
        cleaned_tokens_with_index = [
            (i, token)
            for i, token in enumerate(tokens)
            if len(set(token).intersection({"<", ">", "/", "]", "["})) == 0
        ]
        pos_result = nltk.pos_tag([ct[1] for ct in cleaned_tokens_with_index])
        result = [[token, ","] for token in tokens]
        for (i, _), (token, pos) in zip(cleaned_tokens_with_index, pos_result):
            _, individual_pos = nltk.pos_tag([token])[0]
            if individual_pos.startswith("NN") or individual_pos.startswith("VB"):
                result[i][1] = individual_pos
            else:
                result[i][1] = pos
        return [tuple(r) for r in result]
