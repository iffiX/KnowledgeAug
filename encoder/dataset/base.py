import torch as t
import time
from typing import Callable, Dict, List, Union, Any
from torch.utils.data import Dataset, IterableDataset
from transformers import BatchEncoding

import traceback


class StaticMapDataset(Dataset):
    def __init__(self, encodings: Union[Dict, BatchEncoding]):
        self.encodings = encodings
        self.override_hook = None

    def set_override_hook(self, override_hook):
        self.override_hook = override_hook

    def __getitem__(self, idx: int):
        result = {
            key: t.tensor(val[idx]).unsqueeze(0) for key, val in self.encodings.items()
        }
        if self.override_hook is not None:
            result = self.override_hook(result)
        return result

    def __len__(self):
        return len(self.encodings["input_ids"])


class StaticIterableDataset(Dataset):
    def __init__(
        self,
        length: int,
        generator: Callable[..., Dict[str, t.Tensor]],
        generator_args: tuple = (),
    ):
        self.length = length
        self.generator = generator
        self.generator_args = generator_args
        self.override_hook = None

    def set_override_hook(self, override_hook):
        self.override_hook = override_hook

    def __getitem__(self, idx: int):
        result = self.generator(idx, *self.generator_args)
        if self.override_hook is not None:
            result = self.override_hook(result)
        return result

    def __len__(self):
        return self.length


class DynamicIterableDataset(IterableDataset):
    def __init__(
        self,
        generator: Callable[..., Dict[str, t.Tensor]],
        generator_args: tuple = (),
        seed_setter: Callable[..., None] = None,
    ):
        self.generator = generator
        self.generator_args = generator_args
        self.seed_setter = seed_setter

    def __iter__(self):
        if self.seed_setter is not None:
            self.seed_setter()
        return self

    def __next__(self):
        try:
            result = self.generator(*self.generator_args)
        except StopIteration:
            traceback.print_exc()
            raise ValueError(
                "The generator thrown a StopIteration exception, shouldn't happen here."
            )
        return result


class EmptyDataset(Dataset):
    def __getitem__(self, idx: int):
        return 0

    def __len__(self):
        return 0


class SizeOnePlaceholderDataset(Dataset):
    def __getitem__(self, idx: int):
        return 0

    def __len__(self):
        return 1


class MovableList(list):
    def to(self, *args, **kwargs):
        return self


def collate_function_dict_to_batch_encoding(
    samples: List[Union[BatchEncoding, Dict[str, Any]]]
) -> BatchEncoding:
    assert isinstance(samples, list)
    assert len(samples) > 0
    keys = set(samples[0].keys())
    for other_sample in samples[1:]:
        other_keys = set(other_sample.keys())
        if not other_keys == keys:
            raise ValueError(f"Keys are different: {keys} and {other_keys}")

    result = {}
    for k in keys:
        data_list = MovableList()
        data_list.extend([s[k] for s in samples])
        try:
            if t.is_tensor(data_list[0]):
                result[k] = t.cat(data_list, dim=0)
            elif isinstance(data_list[0], int):
                result[k] = t.tensor(data_list, dtype=t.int64)
            elif isinstance(data_list[0], float):
                result[k] = t.tensor(data_list, dtype=t.float32)
            elif isinstance(data_list[0], MovableList):
                new_data_list = MovableList()
                for item in data_list:
                    for sub_item in item:
                        new_data_list.append(sub_item)
                result[k] = new_data_list
            else:
                result[k] = data_list
        except Exception as e:
            print(f"\nKey:{k}\nInput:\n{data_list}")
            raise e
    return BatchEncoding(data=result)


def dict_iter(samples: Union[BatchEncoding, Dict[str, Any]], keep_dimension=True):
    keys = list(samples.keys())
    for i in range(len(samples[keys[0]])):
        result = {}
        for k in keys:
            if t.is_tensor(samples[k]):
                if keep_dimension:
                    result[k] = samples[k][i].unsqueeze(0)
                else:
                    result[k] = samples[k][i]
            elif isinstance(samples[k], MovableList):
                if keep_dimension:
                    result[k] = MovableList()
                    result[k].append(samples[k][i])
                else:
                    result[k] = samples[k][i]
        if isinstance(samples, BatchEncoding):
            yield BatchEncoding(result)
        else:
            yield result
