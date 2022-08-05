import os
import copy
import tqdm
import json
import itertools
import warnings
import torch as t
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
    BatchEncoding,
)
from pytorch_lightning.utilities import rank_zero_only
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)
from encoder.models.multiple_choice.model import Model
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.openbook_qa import OpenBookQADataset
from encoder.dataset.openbook_qa_augment import OpenBookQAAugDataset
from encoder.utils.file import JSONCache, TorchCache
from encoder.utils.config import OpenBookQAAugmentTrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    preprocess_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.adafactor import Adafactor


class OpenBookQAAugmentTrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQAAugmentTrainConfig,
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
            "t5-base" if "t5-" in self.config.base_type else config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.dataset = OpenBookQAAugDataset(
            tokenizer=self.tokenizer,
            augment_contexts=self.load_augment_contexts(),
            max_seq_length=config.max_seq_length,
            output_mode="single" if "t5-" in self.config.base_type else "splitted",
        )
        if "t5-" in self.config.base_type:
            self.model = T5ForConditionalGeneration.from_pretrained(
                config.base_type,
                cache_dir=model_cache_dir,
                proxies=proxies,
                mirror=huggingface_mirror,
                return_dict=True,
                local_files_only=local_files_only,
            )
        else:
            model_configs = config.model_configs or {}
            self.model = Model(config.base_type, 4, **model_configs)
        self._real_device = None
        self.similarity_embedding = None

    @property
    def monitor(self):
        return "test_accuracy"

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
        if self.trainer.stage_mode == "train":
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
        else:
            validate_dataset = self.dataset.validate_dataset
            validate_dataset.length = 1
            # self.dataset.test_data = self.dataset.test_data[339:340]
            test_dataset = self.dataset.test_dataset
            # test_dataset.length = 100
            return [
                DataLoader(
                    dataset=validate_dataset,
                    num_workers=self.config.load_worker_num,
                    prefetch_factor=self.config.load_prefetch_per_worker,
                    batch_size=1,
                    collate_fn=collate_function_dict_to_batch_encoding,
                    worker_init_fn=set_worker_sharing_strategy,
                ),
                DataLoader(
                    dataset=test_dataset,
                    num_workers=self.config.load_worker_num,
                    prefetch_factor=self.config.load_prefetch_per_worker,
                    batch_size=1,
                    collate_fn=collate_function_dict_to_batch_encoding,
                    worker_init_fn=set_worker_sharing_strategy,
                ),
            ]

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

        # if self.trainer.stage_mode == "validate":
        #     # perform this step after model device is changed to accelerator
        #     self.similarity_embedding = self.load_similarity_embedding()
        #     # must use original train data for correspondence
        #     self.dataset.train_data = self.dataset.original_train_data

    def on_validation_start(self):
        if self.trainer.stage_mode == "validate":
            # perform this step after model device is changed to accelerator
            self.similarity_embedding = self.load_similarity_embedding()
            # must use original train data for correspondence
            self.dataset.train_data = self.dataset.original_train_data

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        # if self.trainer.stage_mode == "validate":
        #     return None

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
            result[:, : out.shape[1]] = out.to(device="cpu", dtype=t.float32)
            batch = batch.to("cpu")
            return {
                "batch": batch,
                "result": result,
            }
        else:
            if self.trainer.stage_mode == "train":
                return {
                    "batch": batch.to("cpu"),
                    "result": self.model.predict(
                        input_ids=batch["sentence"].to(self.real_device),
                        attention_mask=batch["mask"].to(self.real_device),
                        token_type_ids=batch["type_ids"].to(self.real_device),
                    ).to(device="cpu", dtype=t.float32),
                }
            else:
                model = copy.deepcopy(self.model)
                model.train()
                # org_optim = self.optimizers().optimizer
                # optim = type(org_optim)(
                #     model.parameters(), lr=7e-6
                # )  # lr=org_optim.defaults["lr"])
                # optim.load_state_dict(org_optim.state_dict())

                optim_cls = getattr(t.optim, self.config.optimizer_class)
                optim = optim_cls(
                    model.parameters(),
                    lr=5e-6,
                    weight_decay=self.config.l2_regularization,
                )

                # pytorch lightning disables grad during validation by default
                with t.enable_grad():
                    similarity = t.sum(
                        self.similarity_embedding[2][batch["id"][0]]
                        * self.similarity_embedding[1],
                        dim=1,
                    )
                    most_similar_train_indices = t.topk(similarity, k=12).indices
                    for i in range(len(most_similar_train_indices)):
                        if similarity[most_similar_train_indices[i]] < 0.60:
                            break

                    # most_similar_train_indices = t.topk(similarity, k=8).indices
                    # i = 8
                    for _ in range(2):
                        if i > 0:
                            most_similar_train_indices = most_similar_train_indices[:i]

                            batch_size = 4
                            for b in tqdm.tqdm(
                                range(0, len(most_similar_train_indices), batch_size)
                            ):
                                train_batch = collate_function_dict_to_batch_encoding(
                                    [
                                        self.dataset.train_dataset[x]
                                        for x in most_similar_train_indices[
                                            b : b + batch_size
                                        ]
                                    ],
                                ).to(self.real_device)
                                optim.zero_grad()
                                loss = (
                                    model(
                                        input_ids=train_batch["sentence"],
                                        attention_mask=train_batch["mask"],
                                        token_type_ids=train_batch["type_ids"],
                                        labels=train_batch["label"],
                                    ).loss
                                    # / 4
                                )
                                loss.backward()
                                optim.step()
                model.eval()
                return {
                    "batch": batch.to("cpu"),
                    "result": model.predict(
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
        for prefix, dataloader_idx in (("val", 0), ("test", 1)):
            batch, result = collate_and_filter_outputs(outputs[dataloader_idx])
            if "t5-" in self.config.base_type:
                metrics = self.dataset.validate_tokens(batch, result)
            else:
                metrics = self.dataset.validate_logits(batch, result)
            for key, value in metrics.items():
                self.log(f"{prefix}_{key}", value, prog_bar=True, sync_dist=True)
            if not self.is_distributed or get_rank() == 0:
                print(f"Validation on {prefix} result:")
                for key, value in metrics.items():
                    print(f"{prefix}_{key}: {value}")
        # if (
        #     self.trainer.stage_mode == "validate"
        #     and (not self.is_distributed or get_rank() == 0)
        #     and not self.trainer.sanity_checking
        # ):
        #     self.trainer.should_stop = True

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
        dataset = OpenBookQADataset(tokenizer=None)
        matcher = dataset.matcher
        with open(
            os.path.join(preprocess_cache_dir, "openbook_qa_sample_result.json"), "r"
        ) as file:
            raw_contexts = json.load(file)
            contexts = {}
            for id, (raw_paths, raw_path_edges) in raw_contexts.items():
                if len(raw_paths) > 0 and len(raw_paths[0]) > 0:
                    raw_paths = [
                        matcher.sub_paths_to_annotations(
                            x, templates="natural", prioritize_original_annotation=True,
                        )[0]
                        for x in raw_path_edges
                    ]

                paths = [", ".join(path) + " # " for path in raw_paths]
                contexts[id] = list(dict.fromkeys(paths))[:4]

        # authoritative train context
        train_contexts = {}
        with JSONCache(
            os.path.join(preprocess_cache_dir, f"openbook_qa_sample_train_result.json"),
            generate_func=self.generate_train_paths,
        ) as cache:
            print(f"Loaded {len(contexts)} contexts")
            data = cache.data
        for id, (raw_paths, raw_path_edges) in data.items():
            if len(raw_paths) > 0:
                raw_paths = dataset.matcher.sub_paths_to_annotations(
                    raw_path_edges,
                    templates="natural",
                    prioritize_original_annotation=True,
                )
                # only the first level (corresponds to sample config)
                train_contexts[id] = [", ".join(raw_paths[0]) + " # "]
            else:
                train_contexts[id] = []

        return contexts, train_contexts

    def load_similarity_embedding(self):
        with TorchCache(
            os.path.join(
                preprocess_cache_dir, f"openbook_qa_sample_similarity_embedding.pt"
            ),
            generate_func=self.generate_similarity_embedding,
        ) as cache:
            return cache.data

    def generate_train_paths(self):
        dataset = OpenBookQADataset(tokenizer=None)
        train_paths = {}
        for d in tqdm.tqdm(dataset.original_train_data):
            result = dataset.matcher.find_shortest_path(
                source_sentence=d["text_question"],
                target_sentence=d["text_answer"],
                intermediate_nodes=d["original_facts"],
                max_depth_for_each_node=2,
            )
            train_paths[d["id"]] = (result[0], result[1])
        return train_paths

    def generate_similarity_embedding(self):
        dataset = OpenBookQADataset(tokenizer=None)
        model_name = "sentence-transformers/all-mpnet-base-v2"
        batch_size = 32
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        ).to(self.real_device)
        texts = []
        ids = []
        for d in (
            dataset.original_train_data
            + dataset.original_validate_data
            + dataset.original_test_data
        ):
            # texts.append(d["text_question"] + " " + d["text_choices"])
            texts.append(d["text_question"])
            ids.append(d["id"])

        sub_embedding_list = []
        for b in tqdm.tqdm(range(0, len(texts), batch_size)):
            batch = tokenizer(
                texts[b : b + batch_size],
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(self.real_device)
            sub_embedding_list.append(
                self.mean_pooling(model(**batch)[0], batch["attention_mask"])
            )
        e = t.cat(sub_embedding_list)
        e = F.normalize(e, p=2, dim=1).to("cpu")
        train_size = len(dataset.original_train_data)
        return (
            ids[:train_size],
            e[:train_size],
            {
                data_id: data_h.unsqueeze(0)
                for data_id, data_h in zip(ids[train_size:], e[train_size:])
            },
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return t.sum(token_embeddings * input_mask_expanded, 1) / t.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
