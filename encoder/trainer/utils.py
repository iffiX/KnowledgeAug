import math
import torch as t
from typing import List, Any
from transformers import get_constant_schedule
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from ..dataset.matcher.base import BaseMatcher, Edge
from ..dataset.base import collate_function_dict_to_batch_encoding, dict_iter


def set_worker_sharing_strategy(_worker_id: int) -> None:
    t.multiprocessing.set_sharing_strategy("file_system")


def collate_and_filter_outputs(outputs, order_by_ids: List[Any] = None):
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

    if order_by_ids:
        list_of_results = []
        lookup_table = {f[0]: f for f in filtered}
        for id_ in order_by_ids:
            list_of_results.append(lookup_table[id_])
    else:
        list_of_results = filtered
    if t.is_tensor(list_of_results[0][2]):
        results = t.cat([lr[2] for lr in list_of_results], dim=0)
    else:
        results = [lr[2] for lr in list_of_results]
    batch = collate_function_dict_to_batch_encoding([lr[1] for lr in list_of_results])
    return batch, results


def join_sub_path_annotations(sub_path_annotations: List[List[str]]):
    # Each str here is an annotation for an edge
    return [", ".join(s) for s in sub_path_annotations]


def filter_augment_parts(
    raw_paths: List[List[str]],
    raw_paths_edges: List[List[List[Edge]]],
    matcher: BaseMatcher,
    augment_use_levels: int,
    augment_method: str,
    augment_use_parts: str,
    recover_edge_from_raw_decoded_paths: bool = False,
):
    if len(raw_paths) > 0 and len(raw_paths[0]) > 0:
        # If there is at least one path and that path is not empty
        raw_paths = [
            join_sub_path_annotations(
                matcher.sub_paths_to_annotations(
                    y[:augment_use_levels],
                    decoded_sub_paths=x[:augment_use_levels]
                    if recover_edge_from_raw_decoded_paths or augment_use_parts != "all"
                    else None,
                    templates=augment_method,
                    use_parts=augment_use_parts,
                    prioritize_original_annotation=True,
                )
            )
            for x, y in zip(raw_paths, raw_paths_edges)
            if len(y) > 0
        ]

    return raw_paths


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
        allow_continue (`bool`, *optional*, defaults to False):
            Whether to reset the scheduler after th cycle
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    Note:
        This modified version supports restart from the beginning if allow_continue is True
    """

    def lr_lambda(current_step):
        if allow_continue:
            current_step = current_step % num_training_steps
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
