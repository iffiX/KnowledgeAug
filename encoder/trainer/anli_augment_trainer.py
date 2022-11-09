import os
import logging
import torch as t
from itertools import chain
from tqdm import tqdm
from multiprocessing import get_context
from torch.utils.data import DataLoader, RandomSampler
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.anli import (
    ANLIBaseDataset,
    ANLIAugmentDataset,
)
from encoder.dataset.matcher.fact_selector import FactSelector
from encoder.utils.file import PickleCache, JSONCache
from encoder.utils.config import ANLIAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .augment_base_trainer import AugmentBaseTrainer
from .utils import set_worker_sharing_strategy


class AugmentGenerator:
    instance = None

    def __init__(self):
        AugmentGenerator.instance = self
        self.dataset = ANLIBaseDataset(tokenizer=None)
        self.queries = []
        self.query_ids = []
        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            id_ = data["id"]
            self.queries.append(data["text_question"] + " " + data["choices"][0])
            self.queries.append(data["text_question"] + " " + data["choices"][1])
            self.query_ids += [id_, id_]

        self.contexts = {}

        texts, masks = self.dataset.matcher.get_atomic_knowledge_text_and_mask()
        text_to_index = {t: idx for idx, t in enumerate(texts)}

        def generate():
            fact_selector = FactSelector(
                self.queries, texts, max_facts=3, inner_batch_size=1024
            )
            return fact_selector.selected_facts

        with PickleCache(
            os.path.join(
                preprocess_cache_dir, "anli_augment_generator_selected_facts.data"
            ),
            generate,
        ) as cache:
            self.selected_facts = cache.data

        selected_facts = sorted(
            list(set([f for facts in self.selected_facts for f in facts]))
        )
        selected_facts_mask = [masks[text_to_index[f]] for f in selected_facts]
        self.dataset.matcher.add_atomic_knowledge(selected_facts, selected_facts_mask)

        logging.info("Finding best paths")

        ctx = get_context("fork")
        with ctx.Pool() as pool:
            for id_, result in tqdm(
                pool.imap_unordered(self.generate_paths, range(len(self.queries) // 2)),
                total=len(self.queries) // 2,
            ):
                self.contexts[id_] = result

        empty_count = 0
        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            if not self.contexts[data["id"]]:
                empty_count += 1
        total = (
            len(self.dataset.original_train_data)
            + len(self.dataset.validate_data)
            + len(self.dataset.test_data)
        )
        logging.info(
            f"{empty_count} contexts are empty, percent {empty_count * 100 / total:.2f} %"
        )

    @staticmethod
    def generate_paths(idx):
        self = AugmentGenerator.instance
        result = []
        for choice_idx in range(2):
            query_idx = idx * 2 + choice_idx
            for fact in self.selected_facts[query_idx]:
                _, raw_path_edges, *__ = self.dataset.matcher.find_shortest_path(
                    source_sentence=self.queries[query_idx],
                    target_sentence="",
                    intermediate_nodes=[fact],
                    max_depth_for_each_node=2,
                )
                raw_paths = self.dataset.matcher.sub_paths_to_annotations(
                    raw_path_edges,
                    templates="standard",
                    prioritize_original_annotation=False,
                )
                # only the first level (corresponds to sample config)
                path = ", ".join([xx for x in raw_paths for xx in x])
                if len(path) == 0:
                    continue
                result.append(path + " # ")
        return self.query_ids[idx * 2], result


class ANLIAugmentTrainer(AugmentBaseTrainer):
    def __init__(
        self,
        config: ANLIAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__(
            2,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )
        with JSONCache(
            os.path.join(preprocess_cache_dir, f"anli_sample_result.json"),
            generate_func=self.load_augment_contexts,
        ) as cache:
            raw_contexts = cache.data
            dataset = ANLIBaseDataset(None)
            train_ids = [data["id"] for data in dataset.train_data]
            train_labels = [data["label"] for data in dataset.train_data]
            train_contexts = {
                id_: raw_contexts[id_][:3] if label == 0 else raw_contexts[id_][3:]
                for id_, label in zip(train_ids, train_labels)
            }
            self.dataset = ANLIAugmentDataset(
                tokenizer=self.tokenizer,
                augment_contexts=(raw_contexts, train_contexts),
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
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
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
        return AugmentGenerator().contexts
