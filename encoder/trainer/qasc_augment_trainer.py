import os
import json
import random
import hashlib
import itertools
import torch as t
from torch.utils.data import DataLoader, RandomSampler
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.qasc import QASCBaseDataset, QASCAugmentDataset
from encoder.dataset.sample import TrainPathGenerator
from encoder.utils.file import JSONCache
from encoder.utils.config import QASCAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .augment_base_trainer import AugmentBaseTrainer
from .utils import set_worker_sharing_strategy, filter_augment_parts


class QASCAugmentTrainer(AugmentBaseTrainer):
    def __init__(
        self,
        config: QASCAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__(
            8,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )
        self.dataset = QASCAugmentDataset(
            tokenizer=self.tokenizer,
            use_augment=self.config.use_augment,
            augment_contexts=self.load_augment_contexts(),
            max_seq_length=config.max_seq_length,
            output_mode="single" if "t5-" in self.config.base_type else "splitted",
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
        return "accuracy"

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
        return DataLoader(
            dataset=self.dataset.validate_dataset,
            # dataset=self.dataset.test_dataset_with_reference,
            **self.dataloader_args,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.test_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )

    def load_augment_contexts(self):
        dataset = QASCBaseDataset(tokenizer=None)
        dataset.matcher.add_qasc_facts(train_and_validate=False)

        if self.config.augment_method == "original_facts":
            contexts = {}
            train_contexts = {}
            for data in dataset.train_data:
                train_contexts[data["id"]] = contexts[data["id"]] = [
                    ", ".join(data["original_facts"]) + " # "
                ]
            for data in itertools.chain(dataset.validate_data, dataset.test_data):
                contexts[data["id"]] = [", ".join(data["original_facts"]) + " # "]
        else:
            contexts = {}
            with open(
                os.path.join(
                    preprocess_cache_dir,
                    f"qasc_sample_result_{self.config.sample_type}.json",
                ),
                "r",
            ) as file:
                raw_contexts = json.load(file)
                for id, (raw_paths, raw_paths_edges, *_) in raw_contexts.items():
                    raw_paths = filter_augment_parts(
                        raw_paths,
                        raw_paths_edges,
                        dataset.matcher,
                        2,
                        self.config.augment_method,
                        self.config.augment_use_parts,
                        True,
                    )

                    paths = [", ".join(path) + " # " for path in raw_paths]
                    if self.config.sample_type == "sc":
                        # Make sure there is enough randomness
                        random.Random(
                            int(hashlib.sha1(id.encode("utf-8")).hexdigest(), 16)
                            & 0xFFFFFFFF
                        ).shuffle(paths)
                    contexts[id] = list(dict.fromkeys(paths))

            # authoritative train context
            train_contexts = {}
            with JSONCache(
                os.path.join(
                    preprocess_cache_dir,
                    f"qasc_sample_train_result_{self.config.sample_type}.json",
                ),
                generate_func=self.generate_train_paths,
            ) as cache:
                print(f"Loaded {len(contexts)} contexts")
                data = cache.data
            for id, (train_path, train_path_edges) in data.items():
                if len(train_path) > 0:
                    if self.config.augment_method == "raw_decode":
                        pass
                    else:
                        train_path = dataset.matcher.sub_paths_to_annotations(
                            train_path_edges,
                            templates=self.config.augment_method,
                            prioritize_original_annotation=True,
                        )
                    train_contexts[id] = [
                        ", ".join([xx for x in train_path for xx in x]) + " # "
                    ]
                else:
                    train_contexts[id] = []

        return contexts, train_contexts

    def generate_train_paths(self):
        dataset = QASCBaseDataset(tokenizer=None)
        dataset.matcher.add_qasc_facts(train_and_validate=False)
        generator = TrainPathGenerator(
            [
                (d["id"], d["text_question"], d["text_answer"], d["original_facts"],)
                for d in dataset.train_data
            ],
            dataset.matcher,
            max_depth=self.config.max_depth,
        )
        return generator.paths
