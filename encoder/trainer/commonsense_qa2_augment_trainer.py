import os
import json
import logging
import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from itertools import chain
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, AutoTokenizer, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from encoder.models.multiple_choice.model import Model
from encoder.models.embedder import Embedder
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.commonsense_qa2 import (
    CommonsenseQA2BaseDataset,
    CommonsenseQA2AugmentDataset,
)
from encoder.prompter.commonsense_qa2 import CommonsenseQA2Prompter
from encoder.searcher.searcher import ScaleSerpSearcher
from encoder.utils.file import JSONCache
from encoder.utils.config import CommonsenseQA2AugmentTrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    preprocess_cache_dir,
    huggingface_mirror,
)
from encoder.utils.adafactor import Adafactor
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)


class CommonsenseQA2AugmentTrainer(pl.LightningModule):
    def __init__(
        self,
        config: CommonsenseQA2AugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")
        fix_missing(config)
        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed

        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-base" if "t5-" in config.base_type else config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )
        with JSONCache(
            os.path.join(preprocess_cache_dir, f"commonsense_qa2_sample_result.json"),
            generate_func=self.load_augment_contexts,
        ) as cache:
            self.dataset = CommonsenseQA2AugmentDataset(
                tokenizer=self.tokenizer,
                augment_contexts=cache.data,
                max_seq_length=config.max_seq_length,
                output_mode="single" if "t5-" in config.base_type else "splitted",
            )
        if "t5-" in config.base_type:
            self.model = T5ForConditionalGeneration.from_pretrained(
                config.base_type,
                cache_dir=model_cache_dir,
                proxies=proxies,
                mirror=huggingface_mirror,
                return_dict=True,
            )
        else:
            model_configs = config.model_configs or {}
            self.model = Model(config.base_type, 5, **model_configs)
        self._real_device = None

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
        return DataLoader(
            dataset=self.dataset.train_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
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

    def on_fit_start(self):
        if "t5-" in self.config.base_type and self.config.device_map is not None:
            if self.is_distributed:
                raise ValueError(
                    "Parallelize T5 model is incompatible with distributed training."
                )
            start_device_id = [k for k, v in self.config.device_map.items() if 0 in v][
                0
            ]
            # replace device property
            self._real_device = f"cuda:{start_device_id}"
            self.model.parallelize(self.config.device_map)
        else:
            self._real_device = None

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        if "t5-" in self.config.base_type:
            # answer shape [batch_size, sequence_length]
            out = self.model(
                input_ids=batch["sentence"].to(self.real_device),
                attention_mask=batch["mask"].to(self.real_device),
                labels=batch["answer"].to(self.real_device),
            )
        else:
            out = self.model(
                input_ids=batch["sentence"].to(self.real_device),
                attention_mask=batch["mask"].to(self.real_device),
                token_type_ids=batch["type_ids"].to(self.real_device),
                labels=batch["label"].to(self.real_device),
            )
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        if "t5-" in self.config.base_type:
            out = self.model.generate(
                batch["sentence"].to(self.real_device),
                max_length=self.config.generate_length,
                attention_mask=batch["mask"].to(self.real_device),
                early_stopping=True,
            )
            result = t.full(
                [out.shape[0], self.config.generate_length], self.tokenizer.pad_token_id
            )
            result[:, : out.shape[1]] = out.to(device="cpu", dtype=t.float32)
            batch = batch.to("cpu")
            return {
                "batch": batch,
                "result": result,
            }
        else:
            return {
                "batch": batch.to("cpu"),
                "result": self.model.predict(
                    input_ids=batch["sentence"].to(self.real_device),
                    attention_mask=batch["mask"].to(self.real_device),
                    token_type_ids=batch["type_ids"].to(self.real_device),
                ).to(device="cpu", dtype=t.float32),
            }

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        batch, result = collate_and_filter_outputs(outputs)
        if "t5-" in self.config.base_type:
            metrics = self.dataset.validate_tokens(batch, result)
        else:
            metrics = self.dataset.validate_logits(batch, result)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        if not self.is_distributed or get_rank() == 0:
            print(f"Validation result:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

    def test_step(self, batch: BatchEncoding, _batch_idx):
        if "t5-" in self.config.base_type:
            out = self.model.generate(
                batch["sentence"].to(self.real_device),
                max_length=self.config.generate_length,
                attention_mask=batch["mask"].to(self.real_device),
                early_stopping=True,
            )
            result = t.full(
                [out.shape[0], self.config.generate_length], self.tokenizer.pad_token_id
            )
            result[:, : out.shape[1]] = out.to(device="cpu", dtype=t.float32)
            batch = batch.to("cpu")
            return {
                "batch": batch,
                "result": result,
            }
        else:
            return {
                "batch": batch.to("cpu"),
                "result": self.model.predict(
                    input_ids=batch["sentence"].to(self.real_device),
                    attention_mask=batch["mask"].to(self.real_device),
                    token_type_ids=batch["type_ids"].to(self.real_device),
                ).to(device="cpu", dtype=t.float32),
            }

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_main_process(gathered_outputs)
        else:
            self.test_on_main_process(outputs)

    @rank_zero_only
    def test_on_main_process(self, outputs):
        _, result = collate_and_filter_outputs(outputs)
        if "t5-" in self.config.base_type:
            self.dataset.generate_test_result_tokens(result, self.stage_result_path)
        else:
            self.dataset.generate_test_result_logits(result, self.stage_result_path)

    def configure_optimizers(self):
        if self.config.optimizer_class == "Adafactor":
            optim = Adafactor(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        else:
            optim_cls = getattr(t.optim, self.config.optimizer_class)
            optim = optim_cls(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        training_steps = (
            len(self.dataset.train_dataset)
            * self.config.epochs
            // self.config.accumulate_grad_batches
        )
        sch = make_scheduler(
            optim,
            self.config.scheduler_warmup_proportion,
            training_steps,
            self.config.scheduler_cycles,
        )
        return (
            [optim],
            [
                {
                    # REQUIRED: The scheduler instance
                    "scheduler": sch,
                    "monitor": self.monitor,
                }
            ],
        )

    def load_augment_contexts(self):
        dataset = CommonsenseQA2BaseDataset(tokenizer=None)
        prompter = CommonsenseQA2Prompter(dataset)
        queries = []
        query_ids = []
        for data in chain(dataset.train_data, dataset.validate_data, dataset.test_data):
            id_ = data["id"]
            hints = prompter.get_search_hints_of_id(id_)[0]
            if len(hints["search"]) > 1:
                for search in hints["search"]:
                    queries.append(" ".join(search))
                    query_ids.append(id_)
            else:
                queries.append(data["text_question"])
                query_ids.append(id_)

        searcher = ScaleSerpSearcher("commonsense_qa2", queries)
        external_knowledge = [ssr for sr in searcher.search_result for ssr in sr]
        allowed_query_ids = [
            id_ for id_, sr in zip(query_ids, searcher.search_result) for _ in sr
        ]
        matcher = dataset.matcher
        matcher.add_external_knowledge(external_knowledge, allowed_query_ids)

        embedder = Embedder()

        contexts = {}
        for id_, facts in tqdm(
            matcher.allowed_facts.items(), total=len(matcher.allowed_facts)
        ):
            hints = prompter.get_search_hints_of_id(id_)[0]
            facts_embeddings = embedder.embed(matcher.allowed_facts[id_])
            contexts[id_] = []
            for starts, search in zip(hints["starts"], hints["search"]):
                search_embedding = embedder.embed([" ".join(search)])
                best_fact_idx = t.argmax(
                    search_embedding * facts_embeddings.transpose(0, 1), dim=1
                ).item()
                best_fact = matcher.allowed_facts[best_fact_idx]

                _, raw_path_edges, *__ = matcher.find_shortest_path(
                    source_sentence=", ".join(starts),
                    target_sentence=", ".join(search),
                    intermediate_nodes=[best_fact],
                    max_depth_for_each_node=2,
                )
                raw_paths = dataset.matcher.sub_paths_to_annotations(
                    raw_path_edges,
                    templates="standard",
                    prioritize_original_annotation=True,
                )
                # only the first level (corresponds to sample config)
                contexts[id_] += [
                    ", ".join([xx for x in raw_paths for xx in x]) + " # "
                ]

        empty_count = 0
        for data in chain(dataset.train_data, dataset.validate_data, dataset.test_data):
            if data["id"] not in contexts:
                contexts[data["id"]] = []
                empty_count += 1
        total = (
            len(dataset.train_data)
            + len(dataset.validate_data)
            + len(dataset.test_data)
        )
        logging.info(
            f"{empty_count} contexts are empty, percent {empty_count * 100 / total:.2f} %"
        )
        return contexts
