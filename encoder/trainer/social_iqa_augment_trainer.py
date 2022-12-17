import os
import json
import torch as t
from torch.utils.data import DataLoader, RandomSampler
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.social_iqa import (
    SocialIQABaseDataset,
    SocialIQAAugmentDataset,
)
from encoder.utils.config import SocialIQAAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .augment_base_trainer import AugmentBaseTrainer
from .utils import set_worker_sharing_strategy, filter_augment_parts


class SocialIQAAugmentTrainer(AugmentBaseTrainer):
    def __init__(
        self,
        config: SocialIQAAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__(
            3,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )
        self.dataset = SocialIQAAugmentDataset(
            tokenizer=self.tokenizer,
            use_augment=self.config.use_augment,
            augment_contexts=self.load_augment_contexts(),
            max_seq_length=config.max_seq_length,
            output_mode="single" if "t5-" in config.base_type else "splitted",
        )
        self.dataloader_args = {
            "num_workers": self.config.load_worker_num,
            "prefetch_factor": self.config.load_prefetch_per_worker,
            "batch_size": self.config.batch_size,
            "collate_fn": collate_function_dict_to_batch_encoding,
            "worker_init_fn": set_worker_sharing_strategy,
        }

    @property
    def monitor(self):
        return "val_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    @property
    def real_device(self):
        return self._real_device or self.device

    def train_dataloader(self):
        gen = t.Generator()
        gen.manual_seed(42)
        return DataLoader(
            dataset=self.dataset.train_dataset,
            sampler=RandomSampler(self.dataset.train_dataset, generator=gen),
            **self.dataloader_args,
        )

    def val_dataloader(self):
        return [
            DataLoader(dataset=self.dataset.validate_dataset, **self.dataloader_args,),
            DataLoader(dataset=self.dataset.test_ref_dataset, **self.dataloader_args,),
        ]

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset.test_dataset, **self.dataloader_args,)

    def load_augment_contexts(self):
        dataset = SocialIQABaseDataset(tokenizer=None)

        if self.config.augment_method not in (
            "raw_decode",
            "standard",
            "standard_no_symbol",
            "natural",
        ):
            raise ValueError(f"Invalid augment method {self.config.augment_method}")

        contexts = {}
        with open(
            os.path.join(preprocess_cache_dir, f"social_iqa_sample_result_sc.json",),
            "r",
        ) as file:
            raw_contexts = json.load(file)
            for id, (raw_paths, raw_paths_edges, answers) in raw_contexts.items():
                raw_paths = filter_augment_parts(
                    raw_paths,
                    raw_paths_edges,
                    dataset.matcher,
                    1,
                    self.config.augment_method,
                    self.config.augment_use_parts,
                    True,
                )

                paths = []
                added_answer = set()
                for path, answer in zip(raw_paths, answers):
                    if answer not in added_answer:
                        # Only use the first path to that answer
                        paths.append(", ".join(path[:1]) + " # ")
                        added_answer.add(answer)
                # paths = [", ".join(path[:1]) + " # " for path in raw_paths]
                contexts[id] = list(dict.fromkeys(paths))

        return contexts
