import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from typing import Dict, Any
from torch.utils.data import DataLoader, Dataset
from torch.distributed import all_gather_object, get_world_size, get_rank
from encoder.utils.config import TestDistributedTrainConfig, fix_missing
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)


class SimpleNumberDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return idx


class TestDistributedTrainer(pl.LightningModule):
    def __init__(
        self,
        config: TestDistributedTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        fix_missing(config)
        self.config = config
        self.model = t.nn.Linear(512, 1)
        self.is_distributed = is_distributed
        self.automatic_optimization = False

    @property
    def monitor(self):
        return "accuracy"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        return DataLoader(
            dataset=SimpleNumberDataset(self.config.train_dataset_size),
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.train_batch_size,
            collate_fn=lambda x: x,
            worker_init_fn=set_worker_sharing_strategy,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=SimpleNumberDataset(self.config.validate_dataset_size),
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.validate_batch_size,
            collate_fn=lambda x: x,
            worker_init_fn=set_worker_sharing_strategy,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=SimpleNumberDataset(self.config.test_dataset_size),
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.test_batch_size,
            collate_fn=lambda x: x,
            worker_init_fn=set_worker_sharing_strategy,
        )

    # noinspection PyTypeChecker
    def training_step(self, batch, batch_idx):
        print(f"Training input {batch} on device {self.device}")
        input = t.ones([len(batch), 512]) * t.tensor(batch).unsqueeze(1)
        loss = self.model(input.to(self.device))
        return loss

    # noinspection PyTypeChecker
    def validation_step(self, batch, batch_idx):
        print(f"Validating input {batch} on device {self.device}")
        self.log("accuracy", 1)
        return {"batch": batch}

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        print(f"Validate outputs: {outputs}")

    def test_step(self, batch, batch_idx):
        print(f"Testing input {batch} on device {self.device}")
        return {"batch": batch}

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_every_process(gathered_outputs)
        else:
            self.test_on_every_process(outputs)

    def test_on_every_process(self, outputs):
        print(f"Test outputs: {outputs}")

    def configure_optimizers(self):
        return t.optim.Adam(self.model.parameters(), lr=1e-5)
