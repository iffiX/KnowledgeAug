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


class RewardPredictorSingleChoiceDatasetCreator:
    instance = None  # type: "RewardPredictorSingleChoiceDatasetCreator"

    def __init__(
        self,
        data: Union[
            List[Tuple[str, str, str, List[str], List[str]]],
            List[Tuple[str, str, str, List[str], List[str], List[str]]],
        ],
        matcher: BaseMatcher,
        max_depth: int = 2,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        wrong_choice: str = "WRONG_CHOICE",
    ):
        """
        Creates a training dataset for reward prediction

        State: Question + (Optional, Context) + Choice + (last explanation)
        Action: Next explanation to chain after current state
        Wrong Actions: Possible paths that could be used

        For wrong choice, predict special path "wrong_choice" as the first
        explanation, and go no further.

            Eg:
            The predictor will give the highest score to
            Question + (Optional, Context) + Choice [SEP] wrong_choice [SEP]

            and low score to
            Question + (Optional, Context) + Choice [SEP] some_path A->B->C [SEP]

        For right choice predict the right path containing specified intermediate nodes to reach

            Eg:
            Suppose reasoning steps is 1:
            The predictor will first give the highest score to
            Question + (Optional, Context) + Choice [SEP] some_path A->B->C [SEP]
            Then give the highest score to
            Question + (Optional, Context) + Choice + some_path A->B->C [SEP] end_of_reasoning [SEP]

        Args:
            data: id, question, correct choice, all choices, intermediate nodes, optional(choice masks),
                intermediate nodes are string names of nodes that the path must go through.
                choice masks are used to exclude unnecessary regions in the choice, eg: stop words etc.
            max_depth: Max depth to search for each step (the length of each sub path).

        Note:
            Context is a special region where you can include additional information without introducing
            start nodes that the matcher will use to create paths. You can specify the necessary context
            in question as:
            "some_question Context: some_context"

            An example of the choice mask:
                For the following choice, if we want to find a path leading to "stinger" and "wasp":
                "a stinger is used for defense by a wasp"
                The the mask would be:
                "--+++++++--------------------------++++"
        """
        self.data = data
        self.matcher = matcher
        self.max_depth = max_depth
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning
        self.wrong_choice = wrong_choice

    @staticmethod
    def get_transitions_for_sample(data_idx):
        try:
            self = RewardPredictorSingleChoiceDatasetCreator.instance
            source_sentence = (
                self.data[data_idx][1]
                if "Context:" not in self.data[data_idx][1]
                else self.data[data_idx][1][: self.data[data_idx][1].find("Context:")]
            )
            target_mask = (
                self.data[data_idx][5][
                    self.data[data_idx][3].index(self.data[data_idx][2])
                ]
                if len(self.data[data_idx]) == 6
                else ""
            )
            result = self.matcher.find_shortest_path(
                source_sentence=source_sentence,
                target_sentence=self.data[data_idx][2],
                target_mask=target_mask,
                intermediate_nodes=self.data[data_idx][4],
                find_target=True,
                max_depth_for_each_node=self.max_depth,
            )
            start_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
                source_sentence, self.data[data_idx][2], target_mask=target_mask,
            )
            transitions = []
            # path for correct choice
            if len(result[0]) > 0:
                state = (
                    "Question: "
                    + self.data[data_idx][1]
                    + " Choice: "
                    + self.data[data_idx][2]
                    + " Explain: "
                )

                # Note, nodes have one more level than annotations and edges
                # The last edge should be "end of reasoning", and all possible choices are excluded
                visited_nodes = []
                for sub_path_annotations, sub_path_edges, level_start_nodes in zip(
                    result[0] + [[self.end_of_reasoning]], result[1] + [[]], result[2],
                ):
                    # collect paths that will be used as negative samples
                    (
                        list_of_neg_sub_path_annotations,
                        _,
                        _,
                        list_of_neg_sub_path_edges,
                    ) = self.find_available_choices(
                        self,
                        visited_nodes,
                        level_start_nodes,
                        target_nodes,
                        self.data[data_idx][0],
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
                        + self.state_delimiter.join(sub_path_annotations)
                        + self.state_delimiter
                    )
                    visited_nodes += [e[0] for e in sub_path_edges] + [
                        e[2] for e in sub_path_edges
                    ]
                    visited_nodes = list(set(visited_nodes))

            # target nodes is just a placeholder here, the value doesn't matter
            (
                list_of_neg_annotations_for_wrong_choices,
                _,
                _,
                _,
            ) = self.find_available_choices(
                self, [], start_nodes, target_nodes, self.data[data_idx][0],
            )
            choice = random.Random(data_idx).choice(
                [c for c in self.data[data_idx][3] if c != self.data[data_idx][2]]
            )
            state = (
                "Question: "
                + self.data[data_idx][1]
                + " Choice: "
                + choice
                + " Explain: "
            )

            transitions.append(
                (state, [self.wrong_choice], list_of_neg_annotations_for_wrong_choices,)
            )
        except Exception as e:
            print(f"Error: {data_idx}")
            traceback.print_exc()
            raise e
        return data_idx, transitions

    @staticmethod
    def initialize_pool(
        data, matcher, max_depth, state_delimiter, end_of_reasoning, wrong_choice
    ):
        RewardPredictorSingleChoiceDatasetCreator.instance = RewardPredictorSingleChoiceDatasetCreator(
            data, matcher, max_depth, state_delimiter, end_of_reasoning, wrong_choice
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            find_target=True,
            max_depth=self.max_depth,
        )


class RewardPredictorSingleChoiceDatasetCreatorWithFilter(
    RewardPredictorSingleChoiceDatasetCreator
):
    """
    Filter composite nodes by their f beta score to current_reached_nodes.
    """

    @staticmethod
    def initialize_pool(
        data, matcher, max_depth, state_delimiter, end_of_reasoning, wrong_choice
    ):
        RewardPredictorSingleChoiceDatasetCreator.instance = RewardPredictorSingleChoiceDatasetCreatorWithFilter(
            data, matcher, max_depth, state_delimiter, end_of_reasoning, wrong_choice
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            max_depth=self.max_depth,
            find_target=True,
            filter_composite_nodes_by_f_beta=True,
            minimum_f_beta=0.35,
        )


class RewardPredictorSingleChoiceDatasetCreatorWithLimitedNodes(
    RewardPredictorSingleChoiceDatasetCreator
):
    """
    Filter composite nodes by specifying their node id.
    """

    @staticmethod
    def initialize_pool(
        data, matcher, max_depth, state_delimiter, end_of_reasoning, wrong_choice
    ):
        RewardPredictorSingleChoiceDatasetCreator.instance = RewardPredictorSingleChoiceDatasetCreatorWithLimitedNodes(
            data, matcher, max_depth, state_delimiter, end_of_reasoning, wrong_choice
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            max_depth=self.max_depth,
            find_target=True,
            allowed_composite_nodes=self.matcher.allowed_composite_nodes[_id],
        )


class RewardPredictorSingleChoiceDataset(Dataset):
    def __init__(
        self,
        name: str,
        data: Union[
            List[Tuple[str, str, str, List[str], List[str]]],
            List[Tuple[str, str, str, List[str], List[str], List[str]]],
        ],
        matcher: BaseMatcher,
        limit_size: int = None,
        limit_neg_transition_num: int = None,
        max_depth: int = 2,
        negative_samples: Union[int, None] = 3,
        negative_shuffle_seed: int = 42,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        wrong_choice: str = "WRONG_CHOICE",
        creator: Any = None,
    ):
        """
        Args:
            name: Dataset name, used to differ cache from one another.
            data: A list of tuples, each one contain:
                (id, question, correct choice, all choices, intermediate nodes, optional(choice masks)),
                intermediate nodes are string names of nodes that the path must go through.
                choice masks are used to exclude unnecessary regions in the choice, eg: stop words etc.
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
        self.wrong_choice = wrong_choice
        self.creator = creator or RewardPredictorSingleChoiceDatasetCreator
        self.pretrain_data = None

    def __len__(self):
        if self.pretrain_data is None:
            self.load()
        return len(self.pretrain_data)

    def __getitem__(self, idx):
        if self.pretrain_data is None:
            self.load()
        if self.negative_samples is not None:
            negative_samples = min(
                self.negative_samples, len(self.pretrain_data[idx][3])
            )
            self.rand.shuffle(self.pretrain_data[idx][3])
            # state, action, wrong actions
            if self.pretrain_data[idx][2] == [
                self.end_of_reasoning
            ] or self.pretrain_data[idx][2] == [self.wrong_choice]:
                return (
                    self.pretrain_data[idx][0],
                    self.pretrain_data[idx][1],
                    self.state_delimiter.join(self.pretrain_data[idx][2]),
                    [
                        self.state_delimiter.join(x)
                        for x in self.pretrain_data[idx][3][:negative_samples]
                    ],
                )
            else:
                return (
                    self.pretrain_data[idx][0],
                    self.pretrain_data[idx][1],
                    self.state_delimiter.join(self.pretrain_data[idx][2]),
                    [
                        self.state_delimiter.join(x)
                        for x in self.pretrain_data[idx][3][: negative_samples - 2]
                    ]
                    + [self.end_of_reasoning]
                    + [self.wrong_choice],
                )
        else:
            if self.pretrain_data[idx][2] == [
                self.end_of_reasoning
            ] or self.pretrain_data[idx][2] == [self.wrong_choice]:
                append = []
            else:
                append = [self.end_of_reasoning] + [self.wrong_choice]
            return (
                self.pretrain_data[idx][0],
                self.pretrain_data[idx][1],
                self.state_delimiter.join(self.pretrain_data[idx][2]),
                [self.state_delimiter.join(x) for x in self.pretrain_data[idx][3]]
                + append,
            )

    def load(self):
        with PickleCache(
            os.path.join(preprocess_cache_dir, f"{self.name}_reward_predictor_sc.data"),
            generate_func=self.get_pretrain_data,
        ) as cache:
            self.pretrain_data = cache.data

        if self.limit_size is not None:
            shuffled_indices = list(range(len(self.pretrain_data[0])))
            self.rand.shuffle(shuffled_indices)
            self.rand.shuffle(self.pretrain_data)
            self.pretrain_data = self.pretrain_data[: self.limit_size]

        for i in range(10):
            print("Some examples of generated single choice reward predictor data:")
            print(f"Example {i}:")
            print(f"State: {self.pretrain_data[i][1]}")
            print(f"Action: {self.pretrain_data[i][2]}")
            print(f"Wrong actions: {self.pretrain_data[i][3][:3]}")
            print("")

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
                self.wrong_choice,
            ),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (idx, transitions) in pool.imap_unordered(
                self.creator.get_transitions_for_sample, range(len(self.data)),
            ):
                pbar.update()

                new_transitions = []
                for trans in transitions:
                    self.rand.shuffle(trans[2])
                    new_transitions.append(
                        (
                            self.data[idx][0],
                            trans[0],
                            trans[1],
                            trans[2][: self.limit_neg_transition_num]
                            if self.limit_neg_transition_num is not None
                            else trans[2],
                        )
                    )
                result.append((idx, new_transitions))
        transitions = [res[1] for res in sorted(result, key=lambda res: res[0])]
        transitions = [ttr for tr in transitions for ttr in tr]
        return transitions


class RewardPredictorMultipleChoiceDatasetCreator:
    instance = None  # type: "RewardPredictorSingleChoiceDatasetCreator"

    def __init__(
        self,
        data: Union[
            List[Tuple[str, str, str, str, List[str]]],
            List[Tuple[str, str, str, str, List[str], str, str]],
        ],
        matcher: BaseMatcher,
        max_depth: int = 2,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
    ):
        """
        Creates a training dataset for reward prediction

        State: Question + (Optional, Context) + All Choices + (last explanation)
        Action: Next explanation to chain after current state
        Wrong Actions: Possible paths that could be used

        Predict the right path containing specified intermediate nodes to reach

            Eg:
            Suppose reasoning steps is 1:
            The predictor will first give the highest score to
            Question + (Optional, Context) + All Choices [SEP] some_path A->B->C [SEP]
            Then give the highest score to
            Question + (Optional, Context) + All Choices + some_path A->B->C [SEP] end_of_reasoning [SEP]

        Args:
            data: id, question, correct choice, all choices, intermediate nodes,
                  optional(choice mask of the correct choice)
                  optional(choice mask of all choices),

                intermediate nodes are string names of nodes that the path must go through.
                choice masks are used to exclude unnecessary regions in the choice, eg: stop words etc.
            max_depth: Max depth to search for each step (the length of each sub path).

        Note:
            Context is a special region where you can include additional information without introducing
            start nodes that the matcher will use to create paths. You can specify the necessary context
            in question as:
            "some_question Context: some_context"

            An example of the choice mask:
                For the following choices, if we want to find a path leading to "wagon", "lot" and "sleeping":
                "flower pots on a wagon, cars that are in a lot, kids that are sleeping soundly"
                The the mask would be:
                "-----------------+++++---------------------+++----------------++++++++--------"
        """
        self.data = data
        self.matcher = matcher
        self.max_depth = max_depth
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning

    @staticmethod
    def get_transitions_for_sample(data_idx):
        try:
            self = RewardPredictorMultipleChoiceDatasetCreator.instance
            source_sentence = (
                self.data[data_idx][1]
                if "Context:" not in self.data[data_idx][1]
                else self.data[data_idx][1][: self.data[data_idx][1].find("Context:")]
            )

            correct_choice_mask, target_mask = (
                (self.data[data_idx][5], self.data[data_idx][6])
                if len(self.data[data_idx]) == 7
                else ("", "")
            )
            result = self.matcher.find_shortest_path(
                source_sentence=source_sentence,
                target_sentence=self.data[data_idx][2],
                target_mask=correct_choice_mask,
                intermediate_nodes=self.data[data_idx][4],
                find_target=True,
                max_depth_for_each_node=self.max_depth,
            )
            start_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
                source_sentence, self.data[data_idx][3], target_mask=target_mask,
            )
            transitions = []
            if len(result[0]) > 0:
                state = (
                    "Question: "
                    + self.data[data_idx][1]
                    + " Choice: "
                    + self.data[data_idx][3]
                    + " Explain: "
                )

                # Note, nodes have one more level than annotations and edges
                # The last edge should be "end of reasoning", and all possible choices are excluded
                visited_nodes = []
                for sub_path_annotations, sub_path_edges, level_start_nodes in zip(
                    result[0] + [[self.end_of_reasoning]], result[1] + [[]], result[2],
                ):
                    # collect paths that will be used as negative samples
                    (
                        list_of_neg_sub_path_annotations,
                        _,
                        _,
                        list_of_neg_sub_path_edges,
                    ) = self.find_available_choices(
                        self,
                        visited_nodes,
                        level_start_nodes,
                        target_nodes,
                        self.data[data_idx][0],
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
                        + self.state_delimiter.join(sub_path_annotations)
                        + self.state_delimiter
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
        RewardPredictorMultipleChoiceDatasetCreator.instance = RewardPredictorMultipleChoiceDatasetCreator(
            data, matcher, max_depth, state_delimiter, end_of_reasoning
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            find_target=True,
            max_depth=self.max_depth,
        )


class RewardPredictorMultipleChoiceDatasetCreatorWithFilter(
    RewardPredictorMultipleChoiceDatasetCreator
):
    """
    Filter composite nodes by their f beta score to current_reached_nodes.
    """

    @staticmethod
    def initialize_pool(data, matcher, max_depth, state_delimiter, end_of_reasoning):
        RewardPredictorMultipleChoiceDatasetCreator.instance = RewardPredictorMultipleChoiceDatasetCreatorWithFilter(
            data, matcher, max_depth, state_delimiter, end_of_reasoning
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            max_depth=self.max_depth,
            find_target=True,
            filter_composite_nodes_by_f_beta=True,
            minimum_f_beta=0.35,
        )


class RewardPredictorMultipleChoiceDatasetCreatorWithLimitedNodes(
    RewardPredictorMultipleChoiceDatasetCreator
):
    """
    Filter composite nodes by specifying their node id.
    """

    @staticmethod
    def initialize_pool(data, matcher, max_depth, state_delimiter, end_of_reasoning):
        RewardPredictorMultipleChoiceDatasetCreator.instance = RewardPredictorMultipleChoiceDatasetCreatorWithLimitedNodes(
            data, matcher, max_depth, state_delimiter, end_of_reasoning
        )

    @staticmethod
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, _id
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=False,
            max_depth=self.max_depth,
            find_target=True,
            allowed_composite_nodes=self.matcher.allowed_composite_nodes[_id],
        )


class RewardPredictorMultipleChoiceDataset(Dataset):
    def __init__(
        self,
        name: str,
        data: Union[
            List[Tuple[str, str, str, str, List[str]]],
            List[Tuple[str, str, str, str, List[str], str, str]],
        ],
        matcher: BaseMatcher,
        limit_size: int = None,
        limit_neg_transition_num: int = None,
        max_depth: int = 2,
        negative_samples: Union[int, None] = 3,
        negative_shuffle_seed: int = 42,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        creator: Any = None,
    ):
        """
        Args:
            name: Dataset name, used to differ cache from one another.
            data: id, question, correct choice, all choices, intermediate nodes,
                  optional(choice mask of the correct choice)
                  optional(choice mask of all choices),

                intermediate nodes are string names of nodes that the path must go through.
                choice masks are used to exclude unnecessary regions in the choice, eg: stop words etc.
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
        self.creator = creator or RewardPredictorMultipleChoiceDatasetCreator
        self.pretrain_data = None

    def __len__(self):
        if self.pretrain_data is None:
            self.load()
        return len(self.pretrain_data)

    def __getitem__(self, idx):
        if self.pretrain_data is None:
            self.load()
        if self.negative_samples is not None:
            negative_samples = min(
                self.negative_samples, len(self.pretrain_data[idx][3])
            )
            self.rand.shuffle(self.pretrain_data[idx][3])
            # state, action, wrong actions
            if self.pretrain_data[idx][2] == [self.end_of_reasoning]:
                return (
                    self.pretrain_data[idx][0],
                    self.pretrain_data[idx][1],
                    self.state_delimiter.join(self.pretrain_data[idx][2]),
                    [
                        self.state_delimiter.join(x)
                        for x in self.pretrain_data[idx][3][:negative_samples]
                    ],
                )
            else:
                return (
                    self.pretrain_data[idx][0],
                    self.pretrain_data[idx][1],
                    self.state_delimiter.join(self.pretrain_data[idx][2]),
                    [
                        self.state_delimiter.join(x)
                        for x in self.pretrain_data[idx][3][: negative_samples - 1]
                    ]
                    + [self.end_of_reasoning],
                )
        else:
            if self.pretrain_data[idx][2] == [self.end_of_reasoning]:
                append = []
            else:
                append = [self.end_of_reasoning]
            return (
                self.pretrain_data[idx][0],
                self.pretrain_data[idx][1],
                self.state_delimiter.join(self.pretrain_data[idx][2]),
                [self.state_delimiter.join(x) for x in self.pretrain_data[idx][3]]
                + append,
            )

    def load(self):
        with PickleCache(
            os.path.join(preprocess_cache_dir, f"{self.name}_reward_predictor_mc.data"),
            generate_func=self.get_pretrain_data,
        ) as cache:
            self.pretrain_data = cache.data

        if self.limit_size is not None:
            shuffled_indices = list(range(len(self.pretrain_data[0])))
            self.rand.shuffle(shuffled_indices)
            self.rand.shuffle(self.pretrain_data)
            self.pretrain_data = self.pretrain_data[: self.limit_size]

        for i in range(10):
            print("Some examples of generated single choice reward predictor data:")
            print(f"Example {i}:")
            print(f"State: {self.pretrain_data[i][1]}")
            print(f"Action: {self.pretrain_data[i][2]}")
            print(f"Wrong actions: {self.pretrain_data[i][3][:3]}")
            print("")

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

                new_transitions = []
                for trans in transitions:
                    self.rand.shuffle(trans[2])
                    new_transitions.append(
                        (
                            self.data[idx][0],
                            trans[0],
                            trans[1],
                            trans[2][: self.limit_neg_transition_num]
                            if self.limit_neg_transition_num is not None
                            else trans[2],
                        )
                    )
                result.append((idx, new_transitions))
        transitions = [res[1] for res in sorted(result, key=lambda res: res[0])]
        transitions = [ttr for tr in transitions for ttr in tr]
        return transitions


class TrainPathCreator:
    instance = None  # type: TrainPathCreator

    def __init__(
        self,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        max_depth: int = 2,
        min_steps: int = 0,
    ):
        self.data = data
        self.matcher = matcher
        self.max_depth = max_depth
        self.min_steps = min_steps

    @staticmethod
    def get_train_path_for_sample(idx):
        self = TrainPathCreator.instance
        result = self.matcher.find_shortest_path(
            source_sentence=self.data[idx][1],
            target_sentence=self.data[idx][2],
            intermediate_nodes=self.data[idx][3],
            find_target=True,
            max_depth_for_each_node=self.max_depth,
            min_levels_before_checking_target_reached=self.min_steps,
        )
        return self.data[idx][0], (result[0], result[1])

    @staticmethod
    def initialize_pool(data, matcher, max_depth, min_levels):
        TrainPathCreator.instance = TrainPathCreator(
            data, matcher, max_depth, min_levels
        )


class TrainPathGenerator:
    def __init__(
        self,
        data: List[Tuple[str, str, str, List[str]]],
        matcher: BaseMatcher,
        max_depth: int = 2,
        min_steps: int = 0,
    ):
        """
        Args:
            data: id, question, answer, list of intermediate facts
        """
        self.data, self.matcher, self.max_depth, self.min_steps = (
            data,
            matcher,
            max_depth,
            min_steps,
        )
        self.paths = self.generate_train_paths()

    def generate_train_paths(self):
        result = {}
        with mp.Pool(
            initializer=TrainPathCreator.initialize_pool,
            initargs=(self.data, self.matcher, self.max_depth, self.min_steps),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (_id, transitions) in pool.imap_unordered(
                TrainPathCreator.get_train_path_for_sample, range(len(self.data)),
            ):
                pbar.update()
                result[_id] = transitions
        return result


class RewardPredictorSingleChoiceBestFirstBeamSearchDataset(Dataset):
    EOS = None

    def __init__(
        self,
        data: Union[
            List[Tuple[str, str, List[str]]],
            List[Tuple[str, str, List[str], List[str]]],
        ],
        predictor,
        matcher: BaseMatcher,
        existing_ids: Set[str] = None,
        max_steps: int = 2,
        max_depth: int = 2,
        expand_choice_num: int = 4,
        beam_size: int = 1,
        return_beam_num: int = 1,
        max_inference_num: int = 50000,
        min_logits: Union[float, None] = None,
        inference_batch_size: int = 128,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        wrong_choice: str = "WRONG_CHOICE",
        parallel: bool = True,
        prune_low_probability_edges: bool = True,
        prune_low_probability_edges_max_ratio: float = 0.7,
        prune_low_probability_edges_min_logits: float = -12,
        stop_when_reaching_target_nodes: bool = True,
    ):
        """
        Args:
            data: A list of tuples of id, question, choices, and optionally match mask for choices.
        """
        assert return_beam_num <= beam_size

        self.data = data
        self.predictor = predictor
        self.matcher = matcher
        self.existing_ids = existing_ids or set()
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.expand_choice_num = expand_choice_num
        self.beam_size = beam_size
        self.return_beam_num = return_beam_num
        self.max_inference_num = max_inference_num
        self.min_logits = min_logits
        self.inference_batch_size = inference_batch_size
        self.state_delimiter = state_delimiter
        self.end_of_reasoning = end_of_reasoning
        self.wrong_choice = wrong_choice
        self.parallel = parallel
        self.prune_low_probability_edges = prune_low_probability_edges
        self.prune_low_probability_edges_max_ratio = (
            prune_low_probability_edges_max_ratio
        )
        self.prune_low_probability_edges_min_logits = (
            prune_low_probability_edges_min_logits
        )
        self.stop_when_reaching_target_nodes = stop_when_reaching_target_nodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, data_idx):
        try:
            print(f"Processing idx = {data_idx}")
            if (
                self.data[data_idx][0] in self.existing_ids
                and self.data[data_idx][0] != "9-433"
            ):
                print(f"Skipping idx = {data_idx}, id = {self.data[data_idx][0]}")
                return self.data[data_idx][0], None, None, None, 0

            annotation_cache = {}
            prune_set = set()
            all_completed_sequences = []
            all_completed_sequences_edges = []
            all_completed_sequences_choices = []
            all_inferenced_action_num = 0

            choices = self.preselect_choices(data_idx)
            for choice in choices:
                target_mask = (
                    self.data[data_idx][3][self.data[data_idx][2].index(choice)]
                    if len(self.data[data_idx]) == 4
                    else ""
                )
                source_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
                    self.data[data_idx][1]
                    if "Context:" not in self.data[data_idx][1]
                    else self.data[data_idx][1][
                        : self.data[data_idx][1].find("Context:")
                    ],
                    choice,
                    target_mask=target_mask,
                )
                if len(source_nodes) == 0:
                    print("Warning: empty source nodes")
                if len(target_nodes) == 0 and len(target_mask) == 0:
                    print("Warning: empty target nodes")
                target_nodes_set = set(target_nodes)
                visited_nodes = []
                current_reached_nodes = source_nodes
                state = (
                    "Question: "
                    + self.data[data_idx][1]
                    + " Choice: "
                    + choice
                    + " Explain: "
                )

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
                            h,
                            (self.EOS, None),
                            (),
                            self.return_and_increase(push_count),
                        )
                        heapq.heappush(q, h)
                    else:
                        (
                            state,
                            visited_nodes,
                            current_reached_nodes,
                        ) = self.get_additional_data(h)

                        has_reached = (
                            len(
                                set(current_reached_nodes).intersection(
                                    target_nodes_set
                                )
                            )
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
                                    annotations,
                                ) = self.get_annotation_data(
                                    data_idx,
                                    visited_nodes,
                                    current_reached_nodes,
                                    target_nodes,
                                    length,
                                    annotation_cache,
                                )
                            all_masked, log_prob = self.log_prob(
                                state,
                                annotations
                                if annotations[-1] is not self.EOS
                                else annotations[:-1] + [self.end_of_reasoning],
                                prune_set,
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

                            # If it is the last level, only keep the annotation with the highest score
                            # since these are just different but similar annotations leading to the
                            # same choice (a path for target nodes only)
                            # This only applies to the case where there are intermediate nodes
                            if length < self.max_steps - 1 or self.max_steps == 1:
                                optimal_annotations = set(highest_scores.values())
                            else:
                                optimal_annotations = {int(t.argmax(log_prob, dim=0))}
                            score = self.get_score(h)
                            if not all_masked:
                                for action in range(len(annotations)):
                                    if action not in optimal_annotations:
                                        continue
                                    sub_path_nodes = [
                                        edge[0]
                                        for edge in list_of_sub_path_edges[action]
                                    ] + [
                                        edge[2]
                                        for edge in list_of_sub_path_edges[action]
                                    ]

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
                        if not self.is_hypothesis_empty(h):
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
                            h,
                            (self.EOS, None),
                            (),
                            self.return_and_increase(push_count),
                        )
                        if not self.is_hypothesis_empty(h):
                            completed_sequences.append(self.get_sequence(h))
                            completed_sequences_edges.append(self.get_sequence_edges(h))
                all_completed_sequences += completed_sequences
                all_completed_sequences_edges += completed_sequences_edges
                all_completed_sequences_choices += [choice] * len(completed_sequences)
                all_inferenced_action_num += inferenced_action_num
            return (
                self.data[data_idx][0],
                all_completed_sequences,
                all_completed_sequences_edges,
                all_completed_sequences_choices,
                all_inferenced_action_num,
            )
        except:
            traceback.print_exc()
            print(f"Skipping idx = {data_idx} because of the exception")

    def preselect_choices(self, data_idx):
        # select choices with lowest wrong choice score
        logits = self.predictor(
            state=[
                (
                    "Question: "
                    + self.data[data_idx][1]
                    + " Choice: "
                    + choice
                    + " Explain: "
                )
                for choice in self.data[data_idx][2]
            ],
            action=[self.wrong_choice] * len(self.data[data_idx][2]),
            inference=True,
            inference_batch_size=self.inference_batch_size,
        )
        return [
            self.data[data_idx][2][i]
            for i in t.topk(
                -logits,
                k=min(self.expand_choice_num, len(self.data[data_idx][2])),
                dim=0,
            ).indices
        ]

    def get_annotation_data(
        self,
        data_idx,
        visited_nodes,
        current_reached_nodes,
        target_nodes,
        length,
        annotation_cache,
    ):
        key = tuple(visited_nodes)
        if key in annotation_cache:
            annotation_data = annotation_cache[key]
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
                self.parallel,
            )

            list_of_sub_path_next_nodes = list_of_sub_path_next_nodes + [[]]
            list_of_sub_path_raw_next_nodes = list_of_sub_path_raw_next_nodes + [-1]
            list_of_sub_path_edges = list_of_sub_path_edges + [[]]

            annotations = [
                self.state_delimiter.join(sub_path_annotations)
                for sub_path_annotations in list_of_sub_path_annotations
            ] + [self.EOS]

            annotation_data = annotation_cache[key] = (
                list_of_sub_path_annotations,
                list_of_sub_path_next_nodes,
                list_of_sub_path_raw_next_nodes,
                list_of_sub_path_edges,
                annotations,
            )
        return annotation_data

    def log_prob(self, state, annotations, prune_set=None, return_top=False):
        if self.prune_low_probability_edges and prune_set is not None:
            allowed_annotations = [
                annotation for annotation in annotations if annotation not in prune_set
            ]
            allowed_annotation_indices = [
                idx
                for idx, annotation in enumerate(annotations)
                if annotation not in prune_set
            ]
            # in case all choices are masked
            if len(allowed_annotations) == 0:
                allowed_annotations = annotations
                allowed_annotation_indices = list(range(len(annotations)))
            partial_logits = self.predictor(
                state=[state] * len(allowed_annotations),
                action=allowed_annotations,
                inference=True,
                inference_batch_size=self.inference_batch_size,
            )
            partial_sorted, partial_indices = t.sort(partial_logits.cpu(), dim=0)
            for s, idx, _ in zip(
                partial_sorted,
                partial_indices,
                range(
                    int(
                        len(allowed_annotations)
                        * self.prune_low_probability_edges_max_ratio
                    )
                ),
            ):
                if s < self.prune_low_probability_edges_min_logits and allowed_annotations[
                    idx
                ] not in (
                    self.end_of_reasoning,
                    self.wrong_choice,
                ):
                    prune_set.add(allowed_annotations[idx])
            logits = t.full(
                [len(annotations), 1],
                -20,
                dtype=partial_logits.dtype,
                device=partial_logits.device,
            )
            logits[allowed_annotation_indices] = partial_logits
        else:
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

        return all_masked, log_prob.cpu()

    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id, parallel
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            find_target=length == self.max_steps - 1,
            max_depth=self.max_depth,
            parallel=parallel,
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
        if len(hypothesis[3][0]) > 0 and hypothesis[3][0][-1][0] is None:
            extend = ()
        else:
            extend = (item,)
        if new_score is not None:
            return (
                -new_score,
                hypothesis[1] + len(extend),
                new_push_count,
                (hypothesis[3][0] + extend, new_additional_data),
            )
        else:
            return (
                hypothesis[0],
                hypothesis[1] + len(extend),
                new_push_count,
                (hypothesis[3][0] + extend, new_additional_data),
            )

    @staticmethod
    def is_hypothesis_ended(hypothesis):
        return len(hypothesis[3][0]) > 0 and hypothesis[3][0][-1][0] is None

    @staticmethod
    def is_hypothesis_empty(hypothesis):
        return len(hypothesis[3][0]) > 0 and hypothesis[3][0][0][0] is None

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


class RewardPredictorSingleChoiceBestFirstBeamSearchDatasetWithFilter(
    RewardPredictorSingleChoiceBestFirstBeamSearchDataset
):
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id, parallel
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=parallel,
            find_target=length == self.max_steps - 1,
            find_composite=length < self.max_steps - 1,
            max_depth=self.max_depth,
            filter_composite_nodes_by_f_beta=True,
            minimum_f_beta=0.35,
        )


class RewardPredictorSingleChoiceBestFirstBeamSearchDatasetWithLimitedNodes(
    RewardPredictorSingleChoiceBestFirstBeamSearchDataset
):
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id, parallel
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=parallel,
            find_target=length == self.max_steps - 1,
            find_composite=length < self.max_steps - 1,
            max_depth=self.max_depth,
            allowed_composite_nodes=self.matcher.allowed_composite_nodes[_id],
        )


class RewardPredictorMultipleChoiceBestFirstBeamSearchDataset(Dataset):
    EOS = None

    def __init__(
        self,
        data: Union[List[Tuple[str, str, str]], List[Tuple[str, str, str, str]]],
        predictor,
        matcher: BaseMatcher,
        existing_ids: Set[str] = None,
        max_steps: int = 2,
        max_depth: int = 2,
        beam_size: int = 5,
        return_beam_num: int = 5,
        max_inference_num: int = 50000,
        min_logits: Union[float, None] = None,
        inference_batch_size: int = 128,
        state_delimiter: str = ", ",
        end_of_reasoning: str = "END_OF_REASONING",
        parallel: bool = True,
        stop_when_reaching_target_nodes: bool = True,
    ):
        """
        Args:
            data: A list of tuples of id, question, all choices, and optionally match mask for all choices.
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
        self.parallel = parallel
        self.stop_when_reaching_target_nodes = stop_when_reaching_target_nodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, data_idx):
        try:
            print(f"Processing idx = {data_idx}")
            if (
                self.data[data_idx][0] in self.existing_ids
                and self.data[data_idx][0] != "9-433"
            ):
                print(f"Skipping idx = {data_idx}, id = {self.data[data_idx][0]}")
                return self.data[data_idx][0], None, None, None, 0

            target_mask = (
                self.data[data_idx][3] if len(self.data[data_idx]) == 4 else ""
            )
            source_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
                self.data[data_idx][1]
                if "Context:" not in self.data[data_idx][1]
                else self.data[data_idx][1][: self.data[data_idx][1].find("Context:")],
                self.data[data_idx][2],
                target_mask=target_mask,
            )
            if len(source_nodes) == 0:
                print("Warning: empty source nodes")
            if len(target_nodes) == 0 and len(target_mask) == 0:
                print("Warning: empty target nodes")
            target_nodes_set = set(target_nodes)
            visited_nodes = []
            current_reached_nodes = source_nodes
            state = (
                "Question: "
                + self.data[data_idx][1]
                + " Choice: "
                + self.data[data_idx][2]
                + " Explain: "
            )

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
                        h, (self.EOS, None), (), self.return_and_increase(push_count),
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
                                annotations,
                            ) = self.get_annotation_data(
                                data_idx,
                                visited_nodes,
                                current_reached_nodes,
                                target_nodes,
                                length,
                            )
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

                        # If it is the last level, only keep the annotation with the highest score
                        # since these are just different but similar annotations leading to the
                        # same choice (a path for target nodes only)
                        # This only applies to the case where there are intermediate nodes
                        if length < self.max_steps - 1 or self.max_steps == 1:
                            optimal_annotations = set(highest_scores.values())
                        else:
                            optimal_annotations = {int(t.argmax(log_prob, dim=0))}
                        score = self.get_score(h)
                        if not all_masked:
                            for action in range(len(annotations)):
                                if action not in optimal_annotations:
                                    continue
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
                    if not self.is_hypothesis_empty(h):
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
                        h, (self.EOS, None), (), self.return_and_increase(push_count),
                    )
                    if not self.is_hypothesis_empty(h):
                        completed_sequences.append(self.get_sequence(h))
                        completed_sequences_edges.append(self.get_sequence_edges(h))
            return (
                self.data[data_idx][0],
                completed_sequences,
                completed_sequences_edges,
                [],  # Make it compatible with single choice beam search outputs
                inferenced_action_num,
            )
        except:
            traceback.print_exc()
            print(f"Skipping idx = {data_idx} because of the exception")

    def get_annotation_data(
        self, data_idx, visited_nodes, current_reached_nodes, target_nodes, length,
    ):
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
            self.parallel,
        )

        list_of_sub_path_next_nodes = list_of_sub_path_next_nodes + [[]]
        list_of_sub_path_raw_next_nodes = list_of_sub_path_raw_next_nodes + [-1]
        list_of_sub_path_edges = list_of_sub_path_edges + [[]]

        annotations = [
            self.state_delimiter.join(sub_path_annotations)
            for sub_path_annotations in list_of_sub_path_annotations
        ] + [self.EOS]

        return (
            list_of_sub_path_annotations,
            list_of_sub_path_next_nodes,
            list_of_sub_path_raw_next_nodes,
            list_of_sub_path_edges,
            annotations,
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

        return all_masked, log_prob.cpu()

    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id, parallel
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            find_target=length == self.max_steps - 1,
            max_depth=self.max_depth,
            parallel=parallel,
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
        if len(hypothesis[3][0]) > 0 and hypothesis[3][0][-1][0] is None:
            extend = ()
        else:
            extend = (item,)
        if new_score is not None:
            return (
                -new_score,
                hypothesis[1] + len(extend),
                new_push_count,
                (hypothesis[3][0] + extend, new_additional_data),
            )
        else:
            return (
                hypothesis[0],
                hypothesis[1] + len(extend),
                new_push_count,
                (hypothesis[3][0] + extend, new_additional_data),
            )

    @staticmethod
    def is_hypothesis_ended(hypothesis):
        return len(hypothesis[3][0]) > 0 and hypothesis[3][0][-1][0] is None

    @staticmethod
    def is_hypothesis_empty(hypothesis):
        return len(hypothesis[3][0]) > 0 and hypothesis[3][0][0][0] is None

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


class RewardPredictorMultipleChoiceBestFirstBeamSearchDatasetWithFilter(
    RewardPredictorMultipleChoiceBestFirstBeamSearchDataset
):
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id, parallel
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=parallel,
            find_target=length == self.max_steps - 1,
            find_composite=length < self.max_steps - 1,
            max_depth=self.max_depth,
            filter_composite_nodes_by_f_beta=True,
            minimum_f_beta=0.35,
        )


class RewardPredictorMultipleChoiceBestFirstBeamSearchDatasetWithLimitedNodes(
    RewardPredictorMultipleChoiceBestFirstBeamSearchDataset
):
    def find_available_choices(
        self, visited_nodes, current_reached_nodes, target_nodes, length, _id, parallel
    ):
        return self.matcher.find_available_choices(
            visited_nodes,
            current_reached_nodes,
            target_nodes,
            parallel=parallel,
            find_target=length == self.max_steps - 1,
            find_composite=length < self.max_steps - 1,
            max_depth=self.max_depth,
            allowed_composite_nodes=self.matcher.allowed_composite_nodes[_id],
        )
