import os
import logging
import torch as t
from torch.distributed import is_initialized, get_rank
from .settings import enable_inspect, inspect_data_dir


def save_inspect_data(data: object, name: str):
    if not is_initialized() or get_rank() == 0:
        if not os.path.exists(inspect_data_dir):
            os.makedirs(inspect_data_dir, exist_ok=True)
        if enable_inspect:
            logging.info(f"[Inspect] Saving {name}")
            t.save(data, os.path.join(inspect_data_dir, f"{name}.data"))
