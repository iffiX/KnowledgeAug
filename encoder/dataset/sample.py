import os
import heapq
import random
import traceback
import multiprocessing as mp
import tqdm
import torch as t
from typing import List, Tuple, Set, Union, Any, Optional
from torch.utils.data import Dataset
from .matcher.base import BaseMatcher
from ..utils.settings import preprocess_cache_dir
from ..utils.file import PickleCache


class RewardPredictorDatasetCreator:
    instance = None  # type: "RewardPredictorDatasetCreator"

    def __init__(
        self,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        max_depth: int = 2,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
    ):
        self.data = data
        self.matcher = matcher
        self.max_depth = max_depth
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning

    @staticmethod
    def get_transitions_for_sample(data_idx):
        try:
            self = RewardPredictorDatasetCreator.instance
            result = self.matcher.find_shortest_path(
                source_sentence=self.data[data_idx][0],
                target_sentence=self.data[data_idx][2],
                intermediate_nodes=self.data[data_idx][3],
                max_depth_for_each_node=self.max_depth,
            )
            _, target_nodes = self.matcher.match_source_and_target_nodes(
                self.data[data_idx][0], self.data[data_idx][2]
            )
            state = self.data[data_idx][0] + " " + self.data[data_idx][1] + " /n "
            transitions = []

            if len(result[0]) > 0:
                # Note, nodes have one more level than annotations and edges
                # The last edge should be "end of reasoning", and all possible choices are excluded
                visited_nodes = []
                for sub_path_annotations, sub_path_edges, start_nodes in zip(
                    result[0] + [[self.end_of_reasoning]], result[1] + [[]], result[2],
                ):
                    # collect paths that will be used as negative samples
                    (
                        list_of_neg_sub_path_annotations,
                        _,
                        _,
                        list_of_neg_sub_path_edges,
                    ) = self.find_available_choices(
                        self, visited_nodes, start_nodes, target_nodes,
                    )

                    # exclude the right sub path
                    list_of_neg_annotations = [
                        neg_sub_path_annotations
                        for neg_sub_path_annotations, neg_sub_path_edges in zip(
                            list_of_neg_sub_path_annotations, list_of_neg_sub_path_edges
                        )
                        if neg_sub_path_edges != sub_path_edges
                    ]
                    # state, correct action, wrong actions
                    transitions.append(
                        (state, sub_path_annotations, list_of_neg_annotations)
                    )
                    state = (
                        state
                        + self.state_delimiter
                        + self.state_delimiter.join(sub_path_annotations)
                    )
                    visited_nodes += [e[0] for e in sub_path_edges] + [
                        e[2] for e in sub_path_edges
                    ]
                    visited_nodes = list(set(visited_nodes))
        except Exception as e:
            print(f"Error: {data_idx}")
            traceback.print_exc()
            raise e
        return data_idx, transitions

    @staticmethod
    def initialize_pool(data, matcher, max_depth, state_delimiter, end_of_reasoning):
        RewardPredictorDatasetCreator.instance = RewardPredictorDatasetCreator(
            data, matcher, max_depth, state_delimiter, end_of_reasoning
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            max_depth=self.max_depth,
        )


class RewardPredictorDatasetCreatorWithFilter(RewardPredictorDatasetCreator):
    @staticmethod
    def initialize_pool(data, matcher, max_depth, state_delimiter, end_of_reasoning):
        RewardPredictorDatasetCreator.instance = RewardPredictorDatasetCreatorWithFilter(
            data, matcher, max_depth, state_delimiter, end_of_reasoning
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            max_depth=self.max_depth,
            filter_composite_nodes_by_f_beta=True,
            minimum_f_beta=0.35,
        )


class RewardPredictorDataset(Dataset):
    def __init__(
        self,
        name: str,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        limit_size: int = None,
        limit_neg_transition_num: int = None,
        max_depth: int = 2,
        negative_samples: Union[int, None] = 5,
        negative_shuffle_seed: int = 42,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        creator=None,
    ):
        """
        Args:
            name: Dataset name, used to differ cache from one another.
            data: A list of tuples of question, choices, answer and facts.
        """
        self.name = name
        self.data = data
        self.rand = random.Random(negative_shuffle_seed)
        self.matcher = matcher
        self.limit_size = limit_size
        self.limit_neg_transition_num = limit_neg_transition_num
        self.max_depth = max_depth
        self.negative_samples = negative_samples
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning
        self.creator = creator or RewardPredictorDatasetCreator
        self.pretrain_data = None

    def __len__(self):
        if self.pretrain_data is None:
            self.load()
        return len(self.pretrain_data[0])

    def __getitem__(self, idx):
        if self.pretrain_data is None:
            self.load()
        if self.negative_samples is not None:
            negative_samples = min(
                self.negative_samples, len(self.pretrain_data[2][idx])
            )
            self.rand.shuffle(self.pretrain_data[2][idx])
            # state, action, wrong actions
            if self.pretrain_data[1][idx] == [self.end_of_reasoning]:
                return (
                    self.pretrain_data[0][idx],
                    self.state_delimiter.join(self.pretrain_data[1][idx]),
                    [
                        self.state_delimiter.join(x)
                        for x in self.pretrain_data[2][idx][:negative_samples]
                    ],
                )
            else:
                return (
                    self.pretrain_data[0][idx],
                    self.state_delimiter.join(self.pretrain_data[1][idx]),
                    [
                        self.state_delimiter.join(x)
                        for x in self.pretrain_data[2][idx][: negative_samples - 1]
                    ]
                    + [self.end_of_reasoning],
                )
        else:
            if self.pretrain_data[1][idx] == [self.end_of_reasoning]:
                append = []
            else:
                append = [self.end_of_reasoning]
            return (
                self.pretrain_data[0][idx],
                self.state_delimiter.join(self.pretrain_data[1][idx]),
                [self.state_delimiter.join(x) for x in self.pretrain_data[2][idx]]
                + append,
            )

    def load(self):
        with PickleCache(
            os.path.join(preprocess_cache_dir, f"{self.name}_reward_predictor.data"),
            generate_func=self.get_pretrain_data,
        ) as cache:
            self.pretrain_data = cache.data

        if self.limit_size is not None:
            shuffled_indices = list(range(len(self.pretrain_data[0])))
            self.rand.shuffle(shuffled_indices)
            self.pretrain_data = (
                [self.pretrain_data[0][i] for i in shuffled_indices[: self.limit_size]],
                [self.pretrain_data[1][i] for i in shuffled_indices[: self.limit_size]],
                [self.pretrain_data[2][i] for i in shuffled_indices[: self.limit_size]],
            )

    def get_pretrain_data(self):
        result = []
        with mp.Pool(
            initializer=self.creator.initialize_pool,
            initargs=(
                self.data,
                self.matcher,
                self.max_depth,
                self.state_delimiter,
                self.end_of_reasoning,
            ),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (idx, transitions) in pool.imap_unordered(
                self.creator.get_transitions_for_sample, range(len(self.data)),
            ):
                pbar.update()
                if self.limit_neg_transition_num is not None:
                    new_transitions = []
                    for trans in transitions:
                        self.rand.shuffle(trans[2])
                        new_transitions.append(
                            (
                                trans[0],
                                trans[1],
                                trans[2][: self.limit_neg_transition_num],
                            )
                        )
                    transitions = new_transitions
                result.append((idx, transitions))
        transitions = [res[1] for res in sorted(result, key=lambda res: res[0])]
        transitions = [ttr for tr in transitions for ttr in tr]
        states = [trans[0] for trans in transitions]
        actions = [trans[1] for trans in transitions]
        wrong_actions = [trans[2] for trans in transitions]
        return states, actions, wrong_actions


class TrainPathCreator:
    instance = None  # type: TrainPathCreator

    def __init__(
        self,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        max_depth: int = 2,
    ):
        self.data = data
        self.matcher = matcher
        self.max_depth = max_depth

    @staticmethod
    def get_train_path_for_sample(idx):
        self = TrainPathCreator.instance
        result = self.matcher.find_shortest_path(
            source_sentence=self.data[idx][1],
            target_sentence=self.data[idx][2],
            intermediate_nodes=self.data[idx][3],
            max_depth_for_each_node=2,
        )
        return self.data[idx][0], (result[0], result[1])

    @staticmethod
    def initialize_pool(data, matcher, max_depth):
        TrainPathCreator.instance = TrainPathCreator(data, matcher, max_depth)


class TrainPathGenerator:
    def __init__(
        self,
        name,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        max_depth: int = 2,
    ):
        self.name = name
        self.data, self.matcher, self.max_depth = data, matcher, max_depth
        self.paths = self.generate_train_paths()

    def generate_train_paths(self):
        result = {}
        with mp.Pool(
            initializer=TrainPathCreator.initialize_pool,
            initargs=(self.data, self.matcher, self.max_depth),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (_id, transitions) in pool.imap_unordered(
                TrainPathCreator.get_train_path_for_sample, range(len(self.data)),
            ):
                pbar.update()
                result[_id] = transitions
        return result


class RewardPredictorBestFirstBeamSearchDataset(Dataset):
    EOS = None

    def __init__(
        self,
        data: List[Tuple[str, str, str]],
        predictor,
        matcher: BaseMatcher,
        existing_ids: Set[str] = None,
        max_steps: int = 2,
        max_depth: int = 2,
        beam_size: int = 20,
        return_beam_num: int = 1,
        max_inference_num: int = 20000,
        min_logits: Union[float, None] = None,
        inference_batch_size: int = 128,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        stop_when_reaching_target_nodes: bool = True,
    ):
        """
        Args:
            data: A list of tuples of id, question, choices.
        """
        assert return_beam_num <= beam_size

        self.data = data
        self.predictor = predictor
        self.matcher = matcher
        self.existing_ids = existing_ids or set()
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.beam_size = beam_size
        self.return_beam_num = return_beam_num
        self.max_inference_num = max_inference_num
        self.min_logits = min_logits
        self.inference_batch_size = inference_batch_size
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning
        self.stop_when_reaching_target_nodes = stop_when_reaching_target_nodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, data_idx):
        try:
            print(f"Processing idx = {data_idx}")
            if self.data[data_idx][0] != "3Y4W8Q93LZJOKV84ZFFFU5C6KYBVDN":
                if self.data[data_idx][0] in self.existing_ids:
                    print(f"Skipping idx = {data_idx}, id = {self.data[data_idx][0]}")
                    return self.data[data_idx][0], None, 0
            source_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
                self.data[data_idx][1], self.data[data_idx][2]
            )
            if len(source_nodes) == 0:
                print("Warning: empty source nodes")
            if len(target_nodes) == 0:
                print("Warning: empty target nodes")
            target_nodes_set = set(target_nodes)
            visited_nodes = []
            current_reached_nodes = source_nodes
            state = self.data[data_idx][1] + " " + self.data[data_idx][2] + " /n "

            # for profiling
            inferenced_action_num = 0

            q = []
            pops = [0] * (self.max_steps + 1)
            push_count = [0]
            heapq.heappush(
                q,
                self.create_hypothesis(
                    0,
                    (),
                    (state, visited_nodes, current_reached_nodes),
                    self.return_and_increase(push_count),
                ),
            )

            completed_sequences = []
            completed_sequences_edges = []

            while (
                not len(completed_sequences) >= self.return_beam_num
                and not len(q) == 0
                and inferenced_action_num < self.max_inference_num
            ):
                h = heapq.heappop(q)

                length = self.get_hypothesis_length(h)
                if length > self.max_steps or pops[length] >= self.beam_size:
                    continue
                pops[length] += 1

                if self.is_hypothesis_ended(h):
                    h = self.extend_hypothesis(
                        h, (self.EOS, None), (), self.return_and_increase(push_count)
                    )
                    heapq.heappush(q, h)
                else:
                    (
                        state,
                        visited_nodes,
                        current_reached_nodes,
                    ) = self.get_additional_data(h)

                    has_reached = (
                        len(set(current_reached_nodes).intersection(target_nodes_set))
                        > 0
                    )

                    if (
                        length > 0
                        and has_reached
                        and self.stop_when_reaching_target_nodes
                    ) or (length > 0 and len(current_reached_nodes) == 1):
                        h = self.extend_hypothesis(
                            h,
                            (self.EOS, None),
                            (),
                            self.return_and_increase(push_count),
                        )
                        heapq.heappush(q, h)
                    else:
                        # if length = 0, only add possible edges
                        # if length > 0 and length < max_steps, add possible edges and EOS
                        # if length == max_steps, only add EOS

                        # And use EOS instead of end_of_reasoning (string comparison) to speed up
                        if length == self.max_steps:
                            annotations = [self.EOS]
                            list_of_sub_path_next_nodes = [[]]
                            list_of_sub_path_raw_next_nodes = [-1]
                            list_of_sub_path_edges = [[]]
                        else:
                            (
                                list_of_sub_path_annotations,
                                list_of_sub_path_next_nodes,
                                list_of_sub_path_raw_next_nodes,
                                list_of_sub_path_edges,
                            ) = self.find_available_choices(
                                visited_nodes,
                                current_reached_nodes,
                                target_nodes,
                                length,
                                self.data[data_idx][0],
                            )
                            annotations = [
                                self.state_delimiter.join(sub_path_annotations)
                                for sub_path_annotations in list_of_sub_path_annotations
                            ]
                            if len(annotations) == 0 and length == 0:
                                print("Warning: empty starting annotations")
                            if len(annotations) == 0 or length > 0:

                                annotations = annotations + [self.EOS]
                                list_of_sub_path_next_nodes = (
                                    list_of_sub_path_next_nodes + [[]]
                                )
                                list_of_sub_path_raw_next_nodes = (
                                    list_of_sub_path_raw_next_nodes + [-1]
                                )
                                list_of_sub_path_edges = list_of_sub_path_edges + [[]]

                        all_masked, log_prob = self.log_prob(
                            state,
                            annotations
                            if annotations[-1] is not self.EOS
                            else annotations[:-1] + [self.end_of_reasoning],
                            return_top=length == 0,
                        )
                        inferenced_action_num += len(annotations)

                        # prune paths with the same end nodes, keep the one with the highest log probability
                        highest_scores = {}
                        for ann_idx, (annotation, next_node) in enumerate(
                            zip(annotations, list_of_sub_path_raw_next_nodes)
                        ):
                            if (
                                next_node not in highest_scores
                                or log_prob[highest_scores[next_node]]
                                < log_prob[ann_idx]
                            ):
                                highest_scores[next_node] = ann_idx

                        suboptimal_paths = t.ones(
                            log_prob.shape, dtype=t.bool, device=log_prob.device
                        )

                        suboptimal_paths[list(highest_scores.values())] = 0
                        log_prob.masked_fill_(suboptimal_paths, -20)

                        score = self.get_score(h)
                        if not all_masked:
                            for action in range(len(annotations)):
                                sub_path_nodes = [
                                    edge[0] for edge in list_of_sub_path_edges[action]
                                ] + [edge[2] for edge in list_of_sub_path_edges[action]]

                                new_score = score + log_prob[action].item()

                                if annotations[action] is self.EOS:
                                    new_additional_data = ()
                                else:
                                    new_state = (
                                        state
                                        + self.state_delimiter
                                        + annotations[action]
                                    )
                                    new_visited_nodes = list(
                                        set(visited_nodes + sub_path_nodes)
                                    )
                                    new_current_reached_nodes = list_of_sub_path_next_nodes[
                                        action
                                    ]
                                    new_additional_data = (
                                        new_state,
                                        new_visited_nodes,
                                        new_current_reached_nodes,
                                    )

                                new_h = self.extend_hypothesis(
                                    h,
                                    (
                                        annotations[action],
                                        tuple(list_of_sub_path_edges[action])
                                        if len(list_of_sub_path_edges[action]) > 0
                                        else None,
                                    ),
                                    new_additional_data,
                                    self.return_and_increase(push_count),
                                    new_score,
                                )

                                heapq.heappush(q, new_h)
                        else:
                            # Assign probability 1 to EOS (log prob = 0)
                            new_h = self.extend_hypothesis(
                                h,
                                (self.EOS, None),
                                (),
                                self.return_and_increase(push_count),
                            )
                            heapq.heappush(q, new_h)

                if self.is_hypothesis_ended(q[0]):
                    h = heapq.heappop(q)
                    completed_sequences.append(self.get_sequence(h))
                    completed_sequences_edges.append(self.get_sequence_edges(h))

            if inferenced_action_num >= self.max_inference_num:
                print(f"idx = {data_idx}, inference bound reached, stopping early")

            if len(completed_sequences) == 0:
                print(f"idx = {data_idx}, no completed sequence, force selection")
                for i in range(self.return_beam_num):
                    if len(q) == 0:
                        break
                    h = heapq.heappop(q)
                    h = self.extend_hypothesis(
                        h, (self.EOS, None), (), self.return_and_increase(push_count)
                    )
                    completed_sequences.append(self.get_sequence(h))
                    completed_sequences_edges.append(self.get_sequence_edges(h))

            return (
                self.data[data_idx][0],
                completed_sequences,
                completed_sequences_edges,
                inferenced_action_num,
            )
        except:
            traceback.print_exc()
            print(f"Skipping idx = {data_idx} because of the exception")

    def log_prob(self, state, annotations, return_top=False):
        logits = self.predictor(
            state=[state] * len(annotations),
            action=annotations,
            inference=True,
            inference_batch_size=self.inference_batch_size,
        )

        all_masked = False
        if self.min_logits is not None:
            logits = logits.squeeze(1)
            mask = logits < self.min_logits
            all_masked = bool(t.all(mask))
            if all_masked and return_top:
                mask = t.ones(logits.shape, dtype=t.bool, device=logits.device)
                mask.index_fill_(
                    0, t.topk(logits, k=min(self.beam_size, logits.shape[0])).indices, 0
                )
                all_masked = False
            exp = t.exp(logits)
            exp[mask] = 0
            softmax = exp / (t.sum(exp) + 1e-10)
            log_prob = t.log(softmax + 1e-10)
        else:
            log_prob = t.log(t.softmax(logits.squeeze(1), dim=0))

        return all_masked, log_prob

    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            max_depth=self.max_depth,
        )

    @staticmethod
    def return_and_increase(counter):
        count = counter[0]
        counter[0] += 1
        return count

    @staticmethod
    def create_hypothesis(score, sequence, additional_data, push_count):
        # heap q put smallest item at front
        # sequence is stored as a tuple of 2-element tuples,
        # the first element is the text annotation (or self.EOS, which is equal to None for END_OF_REASONING)
        # the second element is the tuple of edges (or None for END_OF_REASONING)
        return -score, len(sequence), push_count, (sequence, additional_data)

    @staticmethod
    def extend_hypothesis(
        hypothesis: Tuple[float, int, int, tuple],
        item: Tuple[Union[str, None], Union[tuple, None]],
        new_additional_data: Any,
        new_push_count: int,
        new_score: Optional[float] = None,
    ):
        if (
            len(hypothesis[3][0]) > 1
            and hypothesis[3][0][-1][0] is None
            and item[0] is None
        ):
            extend = ()
        else:
            extend = (item,)
        if new_score is not None:
            return (
                -new_score,
                hypothesis[1] + 1,
                new_push_count,
                (hypothesis[3][0] + extend, new_additional_data),
            )
        else:
            return (
                hypothesis[0],
                hypothesis[1] + 1,
                new_push_count,
                (hypothesis[3][0] + extend, new_additional_data),
            )

    @staticmethod
    def is_hypothesis_ended(hypothesis):
        return len(hypothesis[3][0]) > 0 and hypothesis[3][0][-1][0] is None

    @staticmethod
    def get_hypothesis_length(hypothesis):
        return hypothesis[1]

    @staticmethod
    def get_score(hypothesis):
        return -hypothesis[0]

    @staticmethod
    def get_additional_data(hypothesis):
        return hypothesis[3][1]

    def get_sequence(self, hypothesis):
        return tuple(segment[0] for segment in hypothesis[3][0][:-1])

    @staticmethod
    def get_sequence_edges(hypothesis):
        return tuple(segment[1] for segment in hypothesis[3][0][:-1])

    def __reduce__(self):
        return (
            type(self),
            (
                self.data,
                self.predictor,
                self.matcher,
                self.existing_ids,
                self.max_steps,
                self.max_depth,
                self.beam_size,
                self.return_beam_num,
                self.max_inference_num,
                self.min_logits,
                self.inference_batch_size,
                self.state_delimiter,
                self.end_of_reasoning,
                self.stop_when_reaching_target_nodes,
            ),
        )


class RewardPredictorBestFirstBeamSearchDatasetWithFilter(
    RewardPredictorBestFirstBeamSearchDataset
):
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            max_depth=self.max_depth,
            filter_composite_nodes_by_f_beta=True,
            minimum_f_beta=0.35,
        )
