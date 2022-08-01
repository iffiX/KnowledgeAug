import os
import heapq
import random
import multiprocessing as mp
import tqdm
import torch as t
from typing import List, Tuple, Set, Union
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
        max_steps: int = 5,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
    ):
        self.data = data
        self.matcher = matcher
        self.max_steps = max_steps
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning

    @staticmethod
    def initialize_pool(data, matcher, max_steps, state_delimiter, end_of_reasoning):
        RewardPredictorDatasetCreator.instance = RewardPredictorDatasetCreator(
            data, matcher, max_steps, state_delimiter, end_of_reasoning
        )

    @staticmethod
    def get_transitions_for_sample(idx):
        try:
            self = RewardPredictorDatasetCreator.instance
            result = self.matcher.find_shortest_path(
                source_sentence=self.data[idx][0],
                target_sentence=self.data[idx][2],
                intermediate_nodes=self.data[idx][3],
            )
            _, target_nodes = self.matcher.match_source_and_target_nodes(
                self.data[idx][0], self.data[idx][2]
            )
            state = self.data[idx][0] + " " + self.data[idx][1] + " /n "
            transitions = []

            if len(result[0]) > 0:
                # Note, nodes have one more level than annotations and edges
                # The last edge should be "end of reasoning", and all possible choices are excluded
                visited_nodes = []
                for (
                    idx,
                    (sub_path_annotations, sub_path_edges, start_nodes),
                ) in enumerate(
                    zip(
                        result[0] + [[self.end_of_reasoning]],
                        result[1] + [[]],
                        result[2],
                    )
                ):
                    # collect paths that will be used as negative samples
                    (
                        list_of_neg_sub_path_annotations,
                        _,
                        list_of_neg_sub_path_edges,
                    ) = self.matcher.find_available_choices(
                        visited_nodes, start_nodes, target_nodes
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
            print(f"Error: {idx}")
            raise e
        return idx, transitions


class RewardPredictorDataset(Dataset):
    def __init__(
        self,
        name: str,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        skip: bool = False,
        max_steps: int = 5,
        negative_samples: Union[int, None] = 5,
        negative_shuffle_seed: int = 42,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
    ):
        """
        Args:
            name: Dataset name, used to differ cache from one another.
            data: A list of tuples of question, choices, answer and facts.
        """
        self.data = data
        self.rand = random.Random(negative_shuffle_seed)
        self.matcher = matcher
        self.max_steps = max_steps
        self.negative_samples = negative_samples
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning

        with PickleCache(
            os.path.join(preprocess_cache_dir, f"{name}_reward_predictor.data"),
            generate_func=self.get_pretrain_data,
        ) as cache:
            self.pretrain_data = cache.data

        if skip:
            self.pretrain_data = (
                self.pretrain_data[0][:1],
                self.pretrain_data[1][:1],
                self.pretrain_data[2][:1],
            )

    def __len__(self):
        return len(self.pretrain_data[0])

    def __getitem__(self, idx):
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

    def get_pretrain_data(self):
        result = []
        with mp.Pool(
            initializer=RewardPredictorDatasetCreator.initialize_pool,
            initargs=(
                self.data,
                self.matcher,
                self.max_steps,
                self.state_delimiter,
                self.end_of_reasoning,
            ),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (idx, transitions) in pool.imap_unordered(
                RewardPredictorDatasetCreator.get_transitions_for_sample,
                range(len(self.data)),
            ):
                pbar.update()
                result.append((idx, transitions))
        transitions = [trans[1] for trans in sorted(result, key=lambda trans: trans[0])]
        transitions = [ttr for tr in transitions for ttr in tr]
        states = [trans[0] for trans in transitions]
        actions = [trans[1] for trans in transitions]
        wrong_actions = [trans[2] for trans in transitions]
        return states, actions, wrong_actions


class RewardPredictorBestFirstBeamSearchDataset(Dataset):
    EOS = None

    def __init__(
        self,
        data: List[Tuple[str, str, str]],
        predictor,
        matcher: BaseMatcher,
        existing_ids: Set[str] = None,
        max_steps: int = 5,
        beam_size: int = 20,
        return_beam_num: int = 1,
        max_inference_num: int = 20000,
        min_logits: Union[float, None] = None,
        inference_batch_size: int = 128,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
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
        self.beam_size = beam_size
        self.return_beam_num = return_beam_num
        self.max_inference_num = max_inference_num
        self.min_logits = min_logits
        self.inference_batch_size = inference_batch_size
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print(f"Processing idx = {idx}")
        if self.data[idx][0] in self.existing_ids:
            print(f"Skipping idx = {idx}, id = {self.data[idx][0]}")
            return self.data[idx][0], None, 0
        source_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
            self.data[idx][1], self.data[idx][2]
        )
        if len(source_nodes) == 0:
            print("Warning: empty source nodes")
        if len(target_nodes) == 0:
            print("Warning: empty target nodes")
        target_nodes_set = set(target_nodes)
        visited_nodes = []
        current_reached_nodes = source_nodes
        state = self.data[idx][1] + " " + self.data[idx][2] + " /n "

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
                    h, self.EOS, (), self.return_and_increase(push_count)
                )
                heapq.heappush(q, h)
            else:
                state, visited_nodes, current_reached_nodes = self.get_additional_data(
                    h
                )

                has_reached = (
                    len(set(current_reached_nodes).intersection(target_nodes_set)) > 0
                )

                if length > 0 and has_reached:
                    h = self.extend_hypothesis(
                        h, self.EOS, (), self.return_and_increase(push_count)
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
                        list_of_sub_path_edges = [[]]
                    else:
                        (
                            list_of_sub_path_annotations,
                            list_of_sub_path_next_nodes,
                            list_of_sub_path_edges,
                        ) = self.matcher.find_available_choices(
                            visited_nodes,
                            current_reached_nodes,
                            target_nodes,
                            only_target=length == self.max_steps - 1
                            and self.max_steps > 1,
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
                    for ann_idx, (annotation, next_nodes) in enumerate(
                        zip(annotations, list_of_sub_path_next_nodes)
                    ):
                        next_nodes = tuple(sorted(next_nodes))
                        if next_nodes not in highest_scores:
                            highest_scores[next_nodes] = ann_idx
                        elif log_prob[highest_scores[next_nodes]] < log_prob[ann_idx]:
                            highest_scores[next_nodes] = ann_idx
                    suboptimal_paths = t.ones(
                        log_prob.shape, dtype=t.bool, device=log_prob.device
                    )
                    for ann_idx in highest_scores.values():
                        suboptimal_paths[ann_idx] = 0
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
                                    state + self.state_delimiter + annotations[action]
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
                                annotations[action],
                                new_additional_data,
                                self.return_and_increase(push_count),
                                new_score,
                            )

                            heapq.heappush(q, new_h)
                    else:
                        # Assign probability 1 to EOS (log prob = 0)
                        new_h = self.extend_hypothesis(
                            h, self.EOS, (), self.return_and_increase(push_count)
                        )
                        heapq.heappush(q, new_h)

            if self.is_hypothesis_ended(q[0]):
                h = heapq.heappop(q)
                completed_sequences.append(self.get_sequence(h))

        if inferenced_action_num >= self.max_inference_num:
            print(f"idx = {idx}, inference bound reached, stopping early")

        if len(completed_sequences) == 0:
            print(f"idx = {idx}, no completed sequence, force selection")
            for i in range(self.return_beam_num):
                h = heapq.heappop(q)
                h = self.extend_hypothesis(
                    h, self.EOS, (), self.return_and_increase(push_count)
                )
                completed_sequences.append(self.get_sequence(h))

        return (
            self.data[idx][0],
            completed_sequences,
            inferenced_action_num,
        )

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

    @staticmethod
    def return_and_increase(counter):
        count = counter[0]
        counter[0] += 1
        return count

    @staticmethod
    def create_hypothesis(score, sequence, additional_data, push_count):
        # heap q put smallest item at front
        return -score, len(sequence), push_count, (sequence, additional_data)

    @staticmethod
    def extend_hypothesis(
        hypothesis, item, new_additional_data, new_push_count, new_score=None
    ):
        if len(hypothesis[3][0]) > 1 and hypothesis[3][0][-1] is None and item is None:
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
        return len(hypothesis[3][0]) > 0 and hypothesis[3][0][-1] is None

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
        return tuple(
            h if h is not None else self.end_of_reasoning for h in hypothesis[3][0]
        )
