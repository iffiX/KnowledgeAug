import os
import json
import itertools
import warnings
import torch as t
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, AutoTokenizer, BatchEncoding
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)
from encoder.model.multiple_choice.model import Model
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.commonsense_qa import CommonsenseQADataset
from encoder.utils.config import CommonsenseQATrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    preprocess_cache_dir,
    huggingface_mirror,
)
from encoder.utils.adafactor import Adafactor


class CommonsenseQASearchTrainer(pl.LightningModule):
    def __init__(
        self,
        config: CommonsenseQATrainConfig,
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
        self.dataset = CommonsenseQADataset(
            tokenizer=self.tokenizer,
            max_seq_length=config.max_seq_length,
            generate_length=config.generate_length,
            use_matcher=config.use_matcher,
            matcher_mode=config.matcher_mode,
            matcher_seed=config.seed,
            matcher_config=config.matcher_config,
            include_option_label_in_sentence=config.include_option_label_in_sentence,
            include_option_label_in_answer_and_choices=config.include_option_label_in_answer_and_choices,
            use_option_label_as_answer_and_choices=config.use_option_label_as_answer_and_choices,
            match_closest_when_no_equal=config.match_closest_when_no_equal,
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
        return [
            DataLoader(
                dataset=self.dataset.validate_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                worker_init_fn=set_worker_sharing_strategy,
            ),
            DataLoader(
                dataset=self.dataset.test_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                worker_init_fn=set_worker_sharing_strategy,
            ),
        ]

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
    def validation_step(self, batch: BatchEncoding, _batch_idx, _dataloader_idx):
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
            result[:, : out.shape[1]] = out.cpu()
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
                ).cpu(),
            }

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device or self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        for prefix, dataloader_idx in (("val", 0), ("test", 1)):
            batch, result = collate_and_filter_outputs(outputs[dataloader_idx])
            self.save_targets(prefix, batch, result)
            if dataloader_idx == 0:
                if "t5-" in self.config.base_type:
                    metrics = self.dataset.validate_tokens(batch, result)
                else:
                    metrics = self.dataset.validate_logits(batch, result)

                for key, value in metrics.items():
                    self.log(key, value, prog_bar=True, sync_dist=True)
                if not self.is_distributed or get_rank() == 0:
                    print("Validation result:")
                    for key, value in metrics.items():
                        print(f"{key}: {value}")

    def configure_optimizers(self):
        if self.config.optimizer_class == "Adafactor":
            optim = Adafactor(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        else:
            optim_cls = getattr(t.optim, self.config.optimizer_class)
            optim = optim_cls(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        training_steps = (
            len(self.dataset.train_dataset)
            * self.config.epochs
            // (self.config.batch_size * self.config.accumulate_grad_batches)
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
                    "interval": "step",
                    "monitor": self.monitor,
                }
            ],
        )

    def save_targets(self, split, batch, logits):
        target_dict = {}
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1)
        for i in range(labels.shape[0]):
            choice = batch["choices"][i][labels[i]]
            other_choices = list(set(batch["choices"][i]).difference({choice}))
            targets = [
                choice,
                self.dataset.question_matcher.find_closest_concept(
                    choice, other_choices
                ),
            ]
            target_dict[batch["id"][i]] = targets
        with open(
            os.path.join(preprocess_cache_dir, f"commonsense_qa_{split}_targets.json"),
            "w",
        ) as file:
            json.dump(target_dict, file)
