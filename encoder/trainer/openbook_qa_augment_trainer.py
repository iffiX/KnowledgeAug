import os
import json
import itertools
import torch as t
from torch.utils.data import DataLoader, RandomSampler
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.openbook_qa import OpenBookQABaseDataset, OpenBookQAAugmentDataset
from encoder.dataset.sample import TrainPathGenerator
from encoder.utils.file import JSONCache
from encoder.utils.config import OpenBookQAAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .augment_base_trainer import AugmentBaseTrainer
from .utils import set_worker_sharing_strategy, filter_augment_parts


class OpenBookQAAugmentTrainer(AugmentBaseTrainer):
    def __init__(
        self,
        config: OpenBookQAAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__(
            4,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )
        self.dataset = OpenBookQAAugmentDataset(
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
        return "test_accuracy"

    @property
    def monitor_mode(self):
        return "max"

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
            DataLoader(dataset=self.dataset.validate_dataset, **self.dataloader_args),
            DataLoader(dataset=self.dataset.test_dataset, **self.dataloader_args),
        ]

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset.test_dataset, **self.dataloader_args)

    def load_augment_contexts(self):
        dataset = OpenBookQABaseDataset(tokenizer=None)

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
                    f"openbook_qa_sample_result_{self.config.sample_type}.json",
                ),
                "r",
            ) as file:
                raw_contexts = json.load(file)
                for id_, (raw_paths, raw_paths_edges, *_) in raw_contexts.items():
                    raw_paths = filter_augment_parts(
                        raw_paths,
                        raw_paths_edges,
                        dataset.matcher,
                        1,
                        self.config.augment_method,
                        self.config.augment_use_parts,
                        False,
                    )
                    paths = [", ".join(path) + " # " for path in raw_paths]
                    if self.config.sample_type == "mc":
                        # Only use top 4 facts for multiple choice scenario
                        contexts[id_] = list(dict.fromkeys(paths))[:4]
                    else:
                        contexts[id_] = list(dict.fromkeys(paths))

            # authoritative train context
            train_contexts = {}
            with JSONCache(
                os.path.join(
                    preprocess_cache_dir,
                    f"openbook_qa_sample_train_result_{self.config.sample_type}.json",
                ),
                generate_func=self.generate_train_paths,
            ) as cache:
                print(f"Loaded {len(contexts)} contexts")
                data = cache.data
            for id_, (train_path, train_path_edges) in data.items():
                if len(train_path) > 0:
                    if self.config.augment_method == "raw_decode":
                        pass
                    else:
                        train_path = dataset.matcher.sub_paths_to_annotations(
                            train_path_edges,
                            templates=self.config.augment_method,
                            prioritize_original_annotation=True,
                        )
                    # only use the first level
                    train_contexts[id_] = [", ".join(train_path[0]) + " # "]
                else:
                    train_contexts[id_] = []

            for split_name, split in zip(
                ("train", "validate", "test"),
                (dataset.train_data, dataset.validate_data, dataset.test_data,),
            ):
                total, count = 0, 0
                for data in split:
                    if data["id"] in contexts:
                        total += 1
                        best_length = 0
                        for path in contexts[data["id"]]:
                            length = 0
                            for fact in data["original_facts"]:
                                if fact.replace(" ,", ",") in path:
                                    length += 1
                                else:
                                    break
                            if best_length < length:
                                best_length = length
                        count += best_length
                print(f"Average retrieval length of {split_name}: {count / total}")

        return contexts, train_contexts

    def generate_train_paths(self):
        dataset = OpenBookQABaseDataset(tokenizer=None)
        generator = TrainPathGenerator(
            [
                (d["id"], d["text_question"], d["text_answer"], d["original_facts"],)
                for d in dataset.train_data
            ],
            dataset.matcher,
            max_depth=self.config.max_depth,
        )
        return generator.paths
