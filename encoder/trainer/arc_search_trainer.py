import copy
import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from typing import Dict, Any
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import AutoTokenizer, BatchEncoding
from encoder.model.multiple_choice.model import ModelForRetriever, ModelForReranker
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.arc_search import ARCSearchDataset
from encoder.utils.config import ARCSearchTrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.adafactor import Adafactor
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)


class ARCSearchTrainer(pl.LightningModule):
    def __init__(
        self,
        config: ARCSearchTrainConfig,
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

        self.retriever_tokenizer = AutoTokenizer.from_pretrained(
            config.retriever_base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            config.reranker_base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )

        self.dataset = ARCSearchDataset(
            retriever_tokenizer=self.retriever_tokenizer,
            reranker_tokenizer=self.reranker_tokenizer,
            retriever_negative_samples=config.retriever_negative_samples,
            retriever_max_seq_length=config.retriever_max_seq_length,
            reranker_negative_samples=config.reranker_negative_samples,
            reranker_max_seq_length=config.reranker_max_seq_length,
        )
        retriever_model_configs = config.retriever_model_configs or {}
        self.retriever_model = ModelForRetriever(
            config.retriever_base_type,
            config.retriever_negative_samples + 1,
            **retriever_model_configs,
        )
        reranker_model_configs = config.reranker_model_configs or {}
        self.reranker_model = ModelForReranker(
            config.reranker_base_type,
            config.reranker_negative_samples + 1,
            **reranker_model_configs,
        )
        self.retriever_original_state_dict = copy.deepcopy(
            self.retriever_model.state_dict()
        )
        self.reranker_original_state_dict = copy.deepcopy(
            self.reranker_model.state_dict()
        )
        self.retriever_best_accuracy = 0
        self.retriever_best_top_k = []
        self.reranker_best_accuracy = 0
        self.reranker_current_search_best_accuracy = 0
        self.reranker_saved_search_best_accuracy = 0
        self.reranker_best_choice = []
        self.automatic_optimization = False

    @property
    def monitor(self):
        return "reranker_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        if self.should_train_retriever():
            return DataLoader(
                dataset=self.dataset.train_retriever_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.retriever_batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                worker_init_fn=set_worker_sharing_strategy,
            )
        else:
            return DataLoader(
                dataset=self.dataset.train_reranker_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.reranker_batch_size,
                collate_fn=lambda x: x,
                worker_init_fn=set_worker_sharing_strategy,
            )

    def val_dataloader(self):
        facts_loader = DataLoader(
            dataset=self.dataset.facts_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.retriever_batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        validate_loader = DataLoader(
            dataset=self.dataset.validate_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.retriever_batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        train_candidates_loader = DataLoader(
            dataset=self.dataset.train_retriever_candidates_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.retriever_batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        unannotated_train_loader = DataLoader(
            dataset=self.dataset.unannotated_train_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.retriever_batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        search_loader = DataLoader(
            dataset=self.dataset.search_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.retriever_batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        return [
            facts_loader,
            validate_loader,
            train_candidates_loader,
            unannotated_train_loader,
            search_loader,
        ]

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        retriever_opt, reranker_opt = self.optimizers()
        retriever_sch, reranker_sch = self.lr_schedulers()
        if self.should_train_retriever():
            retriever_out = self.retriever_model(
                input_ids=batch["sentence"].to(self.device),
                attention_mask=batch["mask"].to(self.device),
                token_type_ids=batch["type_ids"].to(self.device)
                if "type_ids" in batch
                else None,
                labels=batch["answer"].to(self.device),
            )
            self.log(
                "retriever_loss", retriever_out.loss.item(), prog_bar=True, on_step=True
            )

            self.manual_backward(retriever_out.loss)
            retriever_sch.step()
            if (batch_idx + 1) % self.config.retriever_accumulate_grad_batches == 0:
                retriever_opt.step()
                retriever_opt.zero_grad()
        else:
            loss = 0
            for data in batch:
                reranker_out = self.reranker_model(
                    input_ids=data["sentence"].to(self.device),
                    attention_mask=data["mask"].to(self.device),
                    token_type_ids=data["type_ids"].to(self.device),
                    labels=t.LongTensor([data["answer"]]).to(self.device),
                )
                loss = loss + reranker_out.loss
            loss = loss / len(batch)
            self.log("reranker_loss", loss.item(), prog_bar=True, on_step=True)

            self.manual_backward(loss)
            reranker_sch.step()
            if (batch_idx + 1) % self.config.reranker_accumulate_grad_batches == 0:
                reranker_opt.step()
                reranker_opt.zero_grad()

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, batch_idx, dataloader_idx):
        if self.trainer.stage_mode == "train":
            if self.should_train_retriever():
                embeds = self.retriever_model.predict_embedding(
                    input_ids=batch["sentence"].to(self.device),
                    attention_mask=batch["mask"].to(self.device),
                    token_type_ids=batch["type_ids"].to(self.device)
                    if "type_ids" in batch
                    else None,
                )
                return {"batch": batch, "result": embeds}
            else:
                return None

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        if self.trainer.stage_mode == "train" and self.should_train_retriever():
            fact_batch, fact_embeds = collate_and_filter_outputs(outputs[0])
            k_num = min(self.config.retriever_top_k, fact_embeds.shape[0])
            validate_batch, validate_embeds = collate_and_filter_outputs(outputs[1])
            validate_top_k = t.topk(
                t.einsum(
                    "vn,fn->vf", (validate_embeds.squeeze(1), fact_embeds.squeeze(1))
                ),
                k=k_num,
                dim=1,
            ).indices

            metrics = self.dataset.validate_search(validate_batch, validate_top_k)
            self.log(
                "retriever_top_k_accuracy",
                metrics["accuracy"],
                prog_bar=True,
                sync_dist=True,
            )
            # Needed by monitor
            self.log(
                "reranker_accuracy",
                self.reranker_best_accuracy,
                prog_bar=True,
                sync_dist=True,
            )
            print("Validation result:")
            print(f"retriever_top_k_accuracy: {metrics['accuracy']}")

            if metrics["accuracy"] > self.retriever_best_accuracy:
                self.retriever_best_accuracy = metrics["accuracy"]

                (
                    train_candidates_batch,
                    train_candidates_embeds,
                ) = collate_and_filter_outputs(outputs[2])
                train_candidates_top_k = t.topk(
                    t.einsum(
                        "tn,fn->tf",
                        (train_candidates_embeds.squeeze(1), fact_embeds.squeeze(1)),
                    ),
                    k=k_num,
                    dim=1,
                ).indices
                self.dataset.set_train_candidate_fact_indices(
                    train_candidates_batch, train_candidates_top_k
                )

                annotate_batch, annotate_embeds = collate_and_filter_outputs(outputs[3])
                annotate_top_k = t.topk(
                    t.einsum(
                        "an,fn->af",
                        (annotate_embeds.squeeze(1), fact_embeds.squeeze(1)),
                    ),
                    k=k_num,
                    dim=1,
                ).indices

                search_batch, search_embeds = collate_and_filter_outputs(outputs[4])
                search_top_k = t.topk(
                    t.einsum(
                        "sn,fn->sf", (search_embeds.squeeze(1), fact_embeds.squeeze(1))
                    ),
                    k=k_num,
                    dim=1,
                ).indices
                self.retriever_best_top_k = [
                    validate_batch,
                    validate_top_k.cpu(),
                    annotate_batch,
                    annotate_top_k.cpu(),
                    search_top_k.cpu(),
                ]
        if self.trainer.stage_mode == "train" and not self.should_train_retriever():
            print("\nGenerating data for validate top-k")
            validate_input = self.move_reranker_input(
                self.dataset.generate_reranker_input(
                    self.retriever_best_top_k[1], "validate"
                )
            )
            print("\nProcessing validate top-k using reranker")
            validate_choice = t.argmax(
                self.reranker_model.predict(*validate_input), dim=1
            )
            # validate_choice = t.topk(
            #     self.reranker_model.predict(*validate_input), k=3, dim=1,
            # ).indices
            metrics = self.dataset.validate_search(
                self.retriever_best_top_k[0],
                self.relative_top_k_to_absolute_fact_index(
                    self.retriever_best_top_k[1], validate_choice
                ),
            )

            self.log(
                "reranker_accuracy", metrics["accuracy"], prog_bar=True, sync_dist=True,
            )
            print("Validation result:")
            print(f"reranker_accuracy: {metrics['accuracy']}")

            if metrics["accuracy"] > self.reranker_best_accuracy:
                self.reranker_best_accuracy = metrics["accuracy"]
                if self.reranker_best_accuracy > 0:
                    print("\nGenerating data for annotate top-k")
                    annotate_input = self.move_reranker_input(
                        self.dataset.generate_reranker_input(
                            self.retriever_best_top_k[3], "unannotated_train"
                        )
                    )
                    print("\nProcessing annotate top-k using reranker")
                    annotate_score = self.reranker_model.predict(*annotate_input)
                    annotate_choice = t.argmax(annotate_score, dim=1)
                    annotate_confidence = (
                        t.softmax(annotate_score, dim=1)[
                            list(range(annotate_choice.shape[0])), annotate_choice
                        ]
                        > 0.5
                    )

                    print("\nGenerating data for search top-k")
                    search_input = self.move_reranker_input(
                        self.dataset.generate_reranker_input(
                            self.retriever_best_top_k[4], "search"
                        )
                    )
                    print("\nProcessing search top-k using reranker")
                    search_candidates = t.topk(
                        self.reranker_model.predict(*search_input), k=5, dim=1,
                    ).indices

                    self.reranker_current_search_best_accuracy = metrics["accuracy"]
                    self.reranker_best_choice = [
                        annotate_choice.cpu(),
                        annotate_confidence.cpu(),
                        search_candidates.cpu(),
                    ]

        if (
            self.reranker_current_search_best_accuracy
            > self.reranker_saved_search_best_accuracy
            or self.trainer.stage_mode == "validate"
        ) and (not self.is_distributed or get_rank() == 0):
            print(
                f"\nSaving train targets, "
                f"best accuracy: {self.reranker_best_accuracy}"
            )
            self.reranker_saved_search_best_accuracy = (
                self.reranker_current_search_best_accuracy
            )
            try:
                self.dataset.save_search_targets(
                    self.relative_top_k_to_absolute_fact_index(
                        self.retriever_best_top_k[4], self.reranker_best_choice[2],
                    )
                )
            except ValueError:
                print("\nSearch targets not saved, ignore this during sanity checking")

        if self.is_end_of_self_train_epoch():
            if len(self.reranker_best_choice) > 0:
                print(
                    f"\nAnnotating train data, "
                    f"best accuracy: {self.reranker_best_accuracy}"
                )
                self.dataset.annotate_train_data(
                    self.retriever_best_top_k[2],
                    self.relative_top_k_to_absolute_fact_index(
                        self.retriever_best_top_k[3], self.reranker_best_choice[0]
                    ),
                    self.reranker_best_choice[1],
                )

            self.retriever_best_accuracy = 0
            self.retriever_best_top_k = []
            self.reranker_best_accuracy = 0
            self.reranker_current_search_best_accuracy = 0
            self.reranker_best_choice = []

    def configure_optimizers(self):
        retriever_opt_cls = (
            getattr(t.optim, self.config.retriever_optimizer_class)
            if self.config.retriever_optimizer_class != "Adafactor"
            else Adafactor
        )
        retriever_opt = retriever_opt_cls(
            self.retriever_model.parameters(),
            lr=self.config.retriever_learning_rate,
            weight_decay=self.config.retriever_l2_regularization,
        )
        retriever_training_steps = (
            len(self.dataset.train_retriever_dataset)
            * self.config.epochs_per_retriever_self_learn
            // (
                self.config.retriever_batch_size
                * self.config.retriever_accumulate_grad_batches
            )
        )
        retriever_sch = make_scheduler(
            retriever_opt,
            self.config.retriever_scheduler_warmup_proportion,
            retriever_training_steps,
            self.config.retriever_scheduler_cycles,
            allow_continue=True,
        )

        reranker_opt_cls = (
            getattr(t.optim, self.config.reranker_optimizer_class)
            if self.config.reranker_optimizer_class != "Adafactor"
            else Adafactor
        )
        reranker_opt = reranker_opt_cls(
            self.reranker_model.parameters(),
            lr=self.config.reranker_learning_rate,
            weight_decay=self.config.reranker_l2_regularization,
        )
        reranker_training_steps = (
            len(self.dataset.train_reranker_dataset)
            * self.config.epochs_per_reranker_self_learn
            // (
                self.config.reranker_batch_size
                * self.config.reranker_accumulate_grad_batches
            )
        )
        reranker_sch = make_scheduler(
            reranker_opt,
            self.config.reranker_scheduler_warmup_proportion,
            reranker_training_steps,
            self.config.reranker_scheduler_cycles,
            allow_continue=True,
        )
        return [retriever_opt, reranker_opt], [retriever_sch, reranker_sch]

    def on_train_epoch_start(self):
        self_learn_epochs = (
            self.config.epochs_per_retriever_self_learn
            + self.config.epochs_per_reranker_self_learn
        )
        if self.current_epoch > 0 and self.current_epoch % self_learn_epochs == 0:
            print("\nReloading original weights")
            self.retriever_model.load_state_dict(self.retriever_original_state_dict)
            self.reranker_model.load_state_dict(self.reranker_original_state_dict)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["retriever_best_accuracy"] = self.retriever_best_accuracy
        checkpoint["retriever_best_top_k"] = self.retriever_best_top_k
        checkpoint["reranker_best_accuracy"] = self.reranker_best_accuracy
        checkpoint[
            "reranker_current_search_best_accuracy"
        ] = self.reranker_current_search_best_accuracy
        checkpoint[
            "reranker_saved_search_best_accuracy"
        ] = self.reranker_saved_search_best_accuracy
        checkpoint["reranker_best_choice"] = self.reranker_best_choice

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.retriever_best_accuracy = checkpoint["retriever_best_accuracy"]
        self.retriever_best_top_k = checkpoint["retriever_best_top_k"]
        self.reranker_best_accuracy = checkpoint["reranker_best_accuracy"]
        self.reranker_current_search_best_accuracy = checkpoint[
            "reranker_current_search_best_accuracy"
        ]
        self.reranker_saved_search_best_accuracy = checkpoint[
            "reranker_saved_search_best_accuracy"
        ]
        self.reranker_best_choice = checkpoint["reranker_best_choice"]

    def should_train_retriever(self):
        self_learn_epochs = (
            self.config.epochs_per_retriever_self_learn
            + self.config.epochs_per_reranker_self_learn
        )
        return (
            self.current_epoch % self_learn_epochs
            < self.config.epochs_per_retriever_self_learn
        )

    def is_end_of_self_train_epoch(self):
        return (self.current_epoch + 1) % (
            self.config.epochs_per_retriever_self_learn
            + self.config.epochs_per_reranker_self_learn
        ) == 0

    def move_reranker_input(self, input):
        return [i.to(self.device) for i in input]

    @staticmethod
    def relative_top_k_to_absolute_fact_index(top_k, top_k_choice):
        if top_k_choice.dim() == 1:
            return top_k[list(range(len(top_k))), top_k_choice]
        else:
            return t.stack(
                [
                    top_k[list(range(len(top_k))), top_k_choice[:, i]]
                    for i in range(top_k_choice.shape[1])
                ],
                dim=1,
            )
