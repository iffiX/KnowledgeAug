import os
import json
import logging
import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, AutoTokenizer, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from encoder.models.multiple_choice.model import Model
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.commonsense_qa import (
    CommonsenseQABaseDataset,
    CommonsenseQAAugmentDataset,
)
from encoder.prompter.commonsense_qa import CommonsenseQAPrompter
from encoder.utils.file import JSONCache
from encoder.utils.config import CommonsenseQAAugmentTrainConfig, fix_missing
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


class CommonsenseQAAugmentTrainer(pl.LightningModule):
    def __init__(
        self,
        config: CommonsenseQAAugmentTrainConfig,
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
        self.dataset = CommonsenseQAAugmentDataset(
            tokenizer=self.tokenizer,
            augment_contexts=self.load_augment_contexts(),
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
        dataset = CommonsenseQABaseDataset(tokenizer=None)
        matcher = dataset.matcher
        with open(
            os.path.join(preprocess_cache_dir, "commonsense_qa_sample_result.json"), "r"
        ) as file:
            raw_contexts = json.load(file)
            contexts = {}
            for id, (raw_paths, raw_path_edges) in raw_contexts.items():
                if len(raw_paths) > 0 and len(raw_paths[0]) > 0:
                    raw_paths = [
                        matcher.sub_paths_to_annotations(
                            x,
                            templates="standard",
                            prioritize_original_annotation=True,
                        )[0]
                        for x in raw_path_edges
                    ]

                paths = [", ".join(path) + " # " for path in raw_paths]
                contexts[id] = list(dict.fromkeys(paths))[:4]

        # authoritative train context
        prompter = CommonsenseQAPrompter(dataset)
        # Add facts used by authoritative reasoning paths
        dataset.matcher.add_question_specific_knowledge(
            prompter.get_all_authoritative_facts()
        )
        train_contexts = {}
        with JSONCache(
            os.path.join(
                preprocess_cache_dir, f"commonsense_qa_sample_train_result.json"
            ),
            generate_func=self.generate_train_paths,
        ) as cache:
            print(f"Loaded {len(contexts)} contexts")
            data = cache.data
        for id, (raw_paths, raw_path_edges) in data.items():
            if len(raw_paths) > 0:
                raw_paths = dataset.matcher.sub_paths_to_annotations(
                    raw_path_edges,
                    templates="standard",
                    prioritize_original_annotation=True,
                )
                # only the first level (corresponds to sample config)
                train_contexts[id] = [", ".join(raw_paths[0]) + " # "]

        return contexts, train_contexts

    def generate_train_paths(self):
        dataset = CommonsenseQABaseDataset(tokenizer=None)
        prompter = CommonsenseQAPrompter(dataset)
        # Add facts used by authoritative reasoning paths
        dataset.matcher.add_question_specific_knowledge(
            prompter.get_all_authoritative_facts()
        )
        train_paths = {}
        for d in tqdm(dataset.train_data):
            if prompter.is_authoritative_reasoning_of_id_available(d["id"]):
                result = dataset.matcher.find_shortest_path(
                    source_sentence=d["text_question"],
                    target_sentence=d["text_answer"],
                    intermediate_nodes=prompter.get_authoritative_reasoning_of_id(
                        d["id"]
                    )[0],
                    max_depth_for_each_node=1,
                )
                train_paths[d["id"]] = (result[0], result[1])
        return train_paths
