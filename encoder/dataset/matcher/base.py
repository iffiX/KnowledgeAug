import nltk
import spacy
import logging
from typing import List, Dict, Tuple, Union
from nltk.corpus import stopwords
from transformers import PreTrainedTokenizerBase
from encoder.dataset.matcher import KnowledgeMatcher

Edge = Tuple[int, int, int, float, str]
Tokens = List[int]


class BaseMatcher:
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
    STOPWORDS_SET = set(stopwords.words("english"))

    def __init__(self, tokenizer: PreTrainedTokenizerBase, matcher: KnowledgeMatcher):
        self.tokenizer = tokenizer
        self.matcher = matcher
        self.nlp = None

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
    ) -> Tuple[List[List[str]], List[List[int]], List[List[Edge]]]:
        """
        Returns:
            A list containing several sub list of edge annotations,
            each sub list corresponding to a sub path.
            A list of next node ids the path is leading to. If the path ends with 
            a composite node, its component nodes are returned.
            A list of sub list of corresponding edges.
        """
        (
            list_of_sub_path_annotation_tokens,
            list_of_sub_path_next_nodes,
            list_of_sub_path_edges,
        ) = self.matcher.find_available_choices(
            visited_nodes, start_nodes, target_nodes, max_depth, only_target
        )
        return (
            [
                [self.tokenizer.decode(tokens) for tokens in sub_path_tokens]
                for sub_path_tokens in list_of_sub_path_annotation_tokens
            ],
            list_of_sub_path_next_nodes,
            list_of_sub_path_edges,
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
        for token, pos in self.safe_pos_tag(tokens):
            token_position = sentence.find(token, offset)
            offset = token_position + len(token)

            # Relaxed matching, If any part is not masked, allow searching for that part
            ids.append(self.tokenizer.encode(token, add_special_tokens=False))
            if (
                (not use_mask or "+" in set(sentence_mask[token_position:offset]))
                and (
                    pos.startswith("NN")
                    or pos.startswith("JJ")
                    or pos.startswith("RB")
                    or (
                        pos.startswith("VB")
                        and token.lower() not in self.VERB_FILTER_SET
                    )
                    or pos.startswith("CD")
                )
                and (not mask_stopwords or token not in self.STOPWORDS_SET)
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
