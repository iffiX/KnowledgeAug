import os
import json
import torch as t
from torch.utils.data import DataLoader, RandomSampler
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.social_iqa import (
    SocialIQABaseDataset,
    SocialIQAAugmentDataset,
)
from encoder.dataset.sample import TrainPathGenerator
from encoder.utils.file import PickleCache, JSONCache
from encoder.utils.config import SocialIQAAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .augment_base_trainer import AugmentBaseTrainer
from .utils import set_worker_sharing_strategy


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

        def flatten_sublists(list):
            return [x for sub_list in list for x in sub_list]

        contexts = {}
        with open(
            os.path.join(preprocess_cache_dir, f"social_iqa_sample_result_mc.json",),
            "r",
        ) as file:
            raw_contexts = json.load(file)
            for id, (raw_paths, raw_path_edges, *_) in raw_contexts.items():
                if self.config.augment_method == "raw_decode":
                    pass
                else:
                    if len(raw_paths) > 0 and len(raw_paths[0]) > 0:
                        raw_paths = [
                            flatten_sublists(
                                dataset.matcher.sub_paths_to_annotations(
                                    x,
                                    decoded_sub_paths=y,
                                    templates="standard",
                                    prioritize_original_annotation=True,
                                )
                            )
                            for x, y in zip(raw_path_edges, raw_paths)
                        ]

                paths = [", ".join(path) + " # " for path in raw_paths][:1]
                contexts[id] = list(dict.fromkeys(paths))

        # authoritative train context
        train_contexts = {}
        with JSONCache(
            os.path.join(
                preprocess_cache_dir, f"social_iqa_sample_train_result_mc.json",
            ),
            generate_func=self.generate_train_paths,
        ) as cache:
            print(f"Loaded {len(contexts)} contexts")
            data = cache.data
        for id, (raw_paths, raw_path_edges) in data.items():
            if len(raw_paths) > 0:
                if self.config.augment_method == "raw_decode":
                    pass
                else:
                    raw_paths = dataset.matcher.sub_paths_to_annotations(
                        raw_path_edges,
                        templates="standard",
                        prioritize_original_annotation=True,
                    )
                train_contexts[id] = [
                    ", ".join([xx for x in raw_paths[:1] for xx in x]) + " # "
                ]
            else:
                train_contexts[id] = []

        return contexts, train_contexts

    def generate_train_paths(self):
        dataset = SocialIQABaseDataset(tokenizer=None)
        generator = TrainPathGenerator(
            [
                (d["id"], d["context"], d["text_answer"], d["facts"],)
                for d in dataset.train_data
            ],
            dataset.matcher,
            max_depth=self.config.max_depth,
            min_steps=1,  # prevent empty results
        )
        return generator.paths
