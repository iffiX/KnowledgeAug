import os
import math
import multiprocessing as mp
import tqdm
import numpy as np
import torch as t
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union
from transformers import get_constant_schedule, PreTrainedTokenizerBase
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from machin.frame.algorithms.gail import Transition, ExpertTransition
from ..dataset.matcher.base import BaseMatcher
from ..dataset.base import collate_function_dict_to_batch_encoding, dict_iter
from ..model.rl.model import Actor, Embedder, MStr
from ..utils.settings import preprocess_cache_dir
from ..utils.file import PickleCache


def set_worker_sharing_strategy(_worker_id: int) -> None:
    t.multiprocessing.set_sharing_strategy("file_system")


def collate_and_filter_outputs(outputs):
    batch = collate_function_dict_to_batch_encoding([o["batch"] for o in outputs])
    if t.is_tensor(outputs[0]["result"]):
        results = t.cat([o["result"] for o in outputs], dim=0)
        list_of_results = [
            (b["id"][0], b, to.unsqueeze(0)) for b, to in zip(dict_iter(batch), results)
        ]
    else:
        results = [r for o in outputs for r in o["result"]]
        list_of_results = [
            (b["id"][0], b, li) for b, li in zip(dict_iter(batch), results)
        ]
    # filter duplicates brought by resetting dataset
    existed = {}
    filtered = []
    for lr in list_of_results:
        if lr[0] not in existed:
            filtered.append(lr)
            existed[lr[0]] = True
    list_of_results = filtered
    if t.is_tensor(list_of_results[0][2]):
        results = t.cat([lr[2] for lr in list_of_results], dim=0)
    else:
        results = [lr[2] for lr in list_of_results]
    batch = collate_function_dict_to_batch_encoding([lr[1] for lr in list_of_results])
    return batch, results


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    last_epoch: int = -1,
    allow_continue: bool = False,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    Note:
        This modified version supports restart from the beginning if allow_continue is True
    """

    def lr_lambda(current_step):
        if allow_continue:
            current_step = current_step % num_warmup_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        if progress >= 1.0:
            return 0.0
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def make_scheduler(
    optimizer, warmup_proportion, training_steps, num_cycles, allow_continue=False
):
    if warmup_proportion <= 0:
        return get_constant_schedule(optimizer)
    else:
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_proportion * training_steps,
            num_training_steps=training_steps,
            num_cycles=num_cycles,
            allow_continue=allow_continue,
        )


class UncheckedTransition(Transition):
    def __init__(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        reward: Union[float, t.Tensor],
        terminal: bool,
        **kwargs,
    ):
        super(UncheckedTransition, self).__init__(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminal=terminal,
            **kwargs,
        )

    def _check_validity(self):
        pass

    def __setitem__(self, key, value):
        object.__setattr__(self, key, value)
        if key not in self._keys:
            self._keys.append(key)
            self.custom_attr.append(key)
        self._check_validity()


class UncheckedExpertTransition(ExpertTransition):
    def __init__(
        self, state: Dict[str, Any], action: Dict[str, Any],
    ):
        super(UncheckedExpertTransition, self).__init__(
            state=state, action=action,
        )

    def _check_validity(self):
        pass


class GAILExpert:
    instance = None

    def __init__(
        self,
        data: List[Tuple[str, str, List[str]]],
        matcher: BaseMatcher,
        state_delimiter: str = ", ",
    ):
        """
        Args:
            data: A list of tuples of question, answer and facts.
        """
        self.data = data
        self.matcher = matcher
        self.state_delimiter = MStr(state_delimiter)

    def get_transitions(self):
        result = []
        with mp.Pool(
            initializer=self.initialize_pool,
            initargs=(self.data, self.matcher, self.state_delimiter),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (idx, transitions) in pool.imap_unordered(
                self.get_transitions_for_sample, range(len(self.data))
            ):
                pbar.update()
                result.append((idx, transitions))
        return [trans[1] for trans in sorted(result, key=lambda trans: trans[0])]

    @staticmethod
    def initialize_pool(data, matcher, state_delimiter):
        GAILExpert.instance = GAILExpert(data, matcher, state_delimiter)

    @staticmethod
    def get_transitions_for_sample(idx):
        self = GAILExpert.instance
        annotations, _edges = self.matcher.find_shortest_path(
            self.data[idx][0], self.data[idx][1], self.data[idx][2]
        )
        state = self.data[idx][0] + "?"
        transitions = []
        for idx, annotation in enumerate(annotations):
            transitions.append(
                {
                    "state": {"state": MStr(state)},
                    "action": {"action_string": MStr(annotation)},
                }
            )
            state = state + self.state_delimiter + annotation
        return idx, transitions


class GAILDataset(Dataset):
    def __init__(
        self,
        data: List[Tuple[str, str, List[str]]],
        matcher: BaseMatcher,
        actor: Actor,
        embedder: Embedder,
        max_steps: int = 5,
        state_delimiter: str = ", ",
    ):
        """
        Args:
            data: A list of tuples of question, answer and facts.
        """
        self.data = data
        self.actor = actor
        self.embedder = embedder
        self.matcher = matcher
        self.max_steps = max_steps
        self.state_delimiter = state_delimiter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_nodes, target_nodes = self.matcher.match_source_and_target_nodes(
            self.data[idx][0], self.data[idx][1]
        )

        target_nodes = set(target_nodes)
        current_reached_nodes = source_nodes
        state = MStr(self.data[idx][0] + "?")
        transitions = []

        with t.no_grad():
            for step in range(self.max_steps):
                annotations, next_nodes, _edges = self.matcher.find_available_choices(
                    current_reached_nodes
                )
                embeddings = self.embedder(annotations)
                action_embedding, *_ = self.actor(state=[state])
                action_index = t.argmax(
                    t.sum(F.normalize(action_embedding, p=2, dim=1) * embeddings, dim=1)
                ).item()
                previous_state = state
                state = state + self.state_delimiter + annotations[action_index]
                current_reached_nodes = [next_nodes[action_index]]
                has_reached = (
                    len(set(current_reached_nodes).intersection(target_nodes)) > 0
                )
                transitions.append(
                    UncheckedTransition(
                        state={"state": MStr(previous_state)},
                        action={
                            "action_embedding": action_embedding,
                            "action_string": MStr(annotations[action_index]),
                        },
                        next_state={"state": MStr(state)},
                        reward=0,
                        terminal=has_reached or step == self.max_steps - 1,
                    )
                )

        return {"transitions": transitions, "intermediate_nodes": self.data[idx][2]}


class GAILActorPretrainDataset(Dataset):
    instance = None

    def __init__(
        self,
        data: List[Tuple[str, str, List[str]]],
        matcher: BaseMatcher,
        embedder: Union[Embedder, None],
        state_delimiter: str = ", ",
        is_sub_process: bool = False,
    ):
        """
        Args:
            data: A list of tuples of question, answer and facts.
        """
        self.data = data
        self.matcher = matcher
        self.embedder = embedder
        self.state_delimiter = state_delimiter
        if not is_sub_process:
            with PickleCache(
                os.path.join(preprocess_cache_dir, "openbook_qa_actor_pretrain.data"),
                generate_func=self.get_pretrain_data,
            ) as cache:
                self.pretrain_data = cache.data

    def __len__(self):
        return len(self.pretrain_data[0])

    def __getitem__(self, idx):
        return {
            "state": self.pretrain_data[0][idx],
            "action_embedding": t.from_numpy(self.pretrain_data[1][idx]).unsqueeze(0),
        }

    def get_pretrain_data(self):
        result = []
        with mp.Pool(
            initializer=self.initialize_pool,
            initargs=(self.data, self.matcher, self.state_delimiter),
        ) as pool, tqdm.tqdm(total=len(self.data)) as pbar:
            for (idx, transitions) in pool.imap_unordered(
                self.get_transitions_for_sample, range(len(self.data))
            ):
                pbar.update()
                result.append((idx, transitions))
        transitions = [trans[1] for trans in sorted(result, key=lambda trans: trans[0])]
        transitions = [ttr for tr in transitions for ttr in tr]
        states = [trans[0] for trans in transitions]
        actions = [trans[1] for trans in transitions]
        with t.no_grad():
            action_embedding = self.embedder(actions).cpu().numpy()
        return states, action_embedding

    @staticmethod
    def initialize_pool(data, matcher, state_delimiter):
        GAILActorPretrainDataset.instance = GAILActorPretrainDataset(
            data, matcher, None, state_delimiter, True
        )

    @staticmethod
    def get_transitions_for_sample(idx):
        self = GAILActorPretrainDataset.instance
        annotations, _edges = self.matcher.find_shortest_path(
            self.data[idx][0], self.data[idx][1], self.data[idx][2]
        )
        state = self.data[idx][0] + "?"
        transitions = []
        for idx, annotation in enumerate(annotations):
            transitions.append((state, annotation))
            state = state + self.state_delimiter + annotation
        return idx, transitions
