import os
import re
import json
import logging
import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from itertools import chain
from tqdm import tqdm
from multiprocessing import get_context
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


class AugmentGenerator:
    instance = None

    def __init__(self):
        AugmentGenerator.instance = self
        self.dataset = CommonsenseQA2BaseDataset(tokenizer=None)
        self.prompter = CommonsenseQA2Prompter(self.dataset)
        queries = []
        query_ids = []
        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            id_ = data["id"]
            hints = self.prompter.get_search_hints_of_id(id_)[0]
            if len(hints["search"]) > 1:
                for search in hints["search"]:
                    queries.append(" ".join(search))
                    query_ids.append(id_)
            else:
                queries.append(data["text_question"])
                query_ids.append(id_)

        searcher = ScaleSerpSearcher("commonsense_qa2", queries)
        search_result = [
            [(key, knowledge.lower()) for key, knowledge in sr]
            for sr in searcher.search_result
        ]
        self.lowered_knowledge_to_knowledge = {
            knowledge.lower(): knowledge
            for sr in searcher.search_result
            for _, knowledge in sr
        }
        external_knowledge = [knowledge for sr in search_result for _, knowledge in sr]
        self.dataset.matcher.add_external_knowledge(external_knowledge)

        embedder = Embedder()

        self.contexts = {}
        self.id_to_searches = {}
        self.id_to_knowledge_keys = {}
        self.knowledge_key_to_knowledge = {}

        for data in chain(
            self.dataset.original_train_data,
            self.dataset.validate_data,
            self.dataset.test_data,
        ):
            id_ = data["id"]
            hints = self.prompter.get_search_hints_of_id(id_)[0]
            self.id_to_searches[id_] = [" ".join(search) for search in hints["search"]]

        for id_, sr in tqdm(zip(query_ids, search_result), total=len(query_ids)):
            if id_ not in self.id_to_knowledge_keys:
                self.id_to_knowledge_keys[id_] = []
            self.id_to_knowledge_keys[id_] += [key for key, _ in sr]
            for key, knowledge in sr:
                self.knowledge_key_to_knowledge[key] = knowledge

        logging.info("Computing embeddings")
        search_embeddings = self.embed_dict(self.id_to_searches, embedder)
        knowledge_keys_embeddings = self.embed_dict(self.id_to_knowledge_keys, embedder)
        self.top_facts_indices = {}
        for id_ in self.id_to_searches:
            if id_ in search_embeddings and id_ in knowledge_keys_embeddings:
                self.top_facts_indices[id_] = (
                    t.topk(
                        t.mm(
                            search_embeddings[id_],
                            knowledge_keys_embeddings[id_].transpose(0, 1),
                        ),
                        k=min(3, knowledge_keys_embeddings[id_].shape[0]),
                        dim=1,
                    )
                    .indices.cpu()
                    .tolist()
                )

        logging.info("Finding best paths")

        ctx = get_context("fork")
        with ctx.Pool() as pool:
            for id_, result in tqdm(
                pool.imap_unordered(self.generate_paths, self.id_to_searches.keys()),
                total=len(self.id_to_searches),
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
    def generate_paths(id_):
        self = AugmentGenerator.instance
        result = []
        if id_ not in self.top_facts_indices:
            return id_, result
        hints = self.prompter.get_search_hints_of_id(id_)[0]
        for starts, search, top in zip(
            hints["starts"], hints["search"], self.top_facts_indices[id_]
        ):
            for idx in top:
                best_knowledge_key = self.id_to_knowledge_keys[id_][idx]
                best_knowledge = self.knowledge_key_to_knowledge[best_knowledge_key]

                _, raw_path_edges, *__ = self.dataset.matcher.find_shortest_path(
                    source_sentence=", ".join(starts),
                    target_sentence=", ".join(search),
                    intermediate_nodes=[best_knowledge],
                    max_depth_for_each_node=2,
                )
                raw_paths = self.dataset.matcher.sub_paths_to_annotations(
                    raw_path_edges,
                    templates="standard",
                    prioritize_original_annotation=False,
                )
                # only the first level (corresponds to sample config)
                path = ", ".join([xx for x in raw_paths for xx in x]).replace(
                    best_knowledge, self.lowered_knowledge_to_knowledge[best_knowledge]
                )
                if len(path) == 0:
                    continue
                result.append(path + " # ")
        return id_, result

    def embed_dict(self, dict, embedder):
        all_strings = []
        all_ids = []
        for id_, strings in dict.items():
            all_strings += strings
            all_ids += [id_] * len(strings)

        embeddings = embedder.embed(all_strings)
        result = {}
        for embed, id_ in zip(embeddings, all_ids):
            if id_ not in result:
                result[id_] = []
            result[id_].append(embed)
        for id_ in result:
            result[id_] = t.stack(result[id_])
        return result


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
            self.model = Model(config.base_type, 2, **model_configs)
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
        return AugmentGenerator().contexts
