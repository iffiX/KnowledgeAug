import os
import json
import itertools
import warnings
import numpy as np
import torch as t
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.distributed import (
    all_gather_object,
    get_world_size,
    get_rank,
    is_initialized,
)
from transformers import BatchEncoding
from sklearn.metrics import accuracy_score, f1_score
from .utils import make_scheduler
from encoder.models.sample.model import RewardPredictor
from encoder.dataset.openbook_qa import OpenBookQADataset
from encoder.dataset.sample import (
    RewardPredictorDataset,
    RewardPredictorBestFirstBeamSearchDataset,
)
from encoder.utils.config import OpenBookQASampleTrainConfig, fix_missing
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.adafactor import Adafactor


class OpenBookQASampleTrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQASampleTrainConfig,
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
        self.dataset = OpenBookQADataset(
            tokenizer=None,
            max_seq_length=config.max_seq_length,
            use_matcher=config.use_matcher,
            matcher_mode=config.matcher_mode,
            matcher_seed=config.seed,
            matcher_config=config.matcher_config,
            match_closest_when_no_equal=config.match_closest_when_no_equal,
            output_mode="single" if "t5-" in self.config.base_type else "splitted",
        )
        self.reward_predictor = RewardPredictor(
            config.base_type,
            pad_by_longest=config.pad_by_longest,
            max_length=config.max_seq_length,
        )
        self.reward_predictor_datasets = {
            split: RewardPredictorDataset(
                f"openbook_qa_{split}",
                [
                    (
                        d["text_question"],
                        d["text_choices"],
                        d["text_answer"],
                        d["original_facts"],
                    )
                    for d in getattr(self.dataset, f"original_{split}_data")
                ],
                self.dataset.matcher,
                limit_size=limit_size,
                max_depth=self.config.max_depth,
                negative_samples=self.config.negative_samples
                if split == "train"
                else None,
                negative_shuffle_seed=self.config.negative_shuffle_seed
                if not is_initialized()
                else self.config.negative_shuffle_seed + get_rank(),
                state_delimiter=self.config.state_delimeter,
                end_of_reasoning=self.config.end_of_reasoning,
            )
            for split, limit_size in (("train", None), ("validate", 1), ("test", 100))
        }
        self.test_result = {}

    @property
    def monitor(self):
        return "test_is_right_max_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        return self.create_sample_dataloader("train")

    def val_dataloader(self):
        return [
            self.create_sample_dataloader("validate"),
            self.create_sample_dataloader("test"),
        ]

    def test_dataloader(self):
        path = os.path.join(preprocess_cache_dir, "openbook_qa_sample_result.json")
        if os.path.exists(path):
            with open(path, "r",) as file:
                self.test_result.update(json.load(file))
            print(f"Existing results num: {len(self.test_result)}")
        return [
            self.create_sample_inference_dataloader("train"),
            self.create_sample_inference_dataloader("validate"),
            self.create_sample_inference_dataloader("test"),
        ]

    # noinspection PyTypeChecker
    def training_step(self, batch, batch_idx):
        states, actions, labels = [], [], []
        for state, action, wrong_actions in batch:
            states += [state] * (1 + len(wrong_actions))
            actions += [action] + wrong_actions
            labels += [1] + [0] * len(wrong_actions)
        labels = t.Tensor(labels).unsqueeze(1).to(self.device)
        loss = nn.BCEWithLogitsLoss(reduction="mean")(
            self.reward_predictor(states, actions), labels,
        )
        return loss

    # noinspection PyTypeChecker
    def validation_step(self, batch, batch_idx, _dataloader_idx):
        # if self.current_epoch < 4:
        #     return {
        #         "accuracy": 0.01 * self.current_epoch,
        #         "f1": 0.01 * self.current_epoch,
        #         "is_right_max": 0.01 * self.current_epoch,
        #     }
        # else:

        state, action, wrong_actions = batch[0]
        states = [state] * (1 + len(wrong_actions))
        actions = [action] + wrong_actions
        labels = np.array([1] + [0] * len(wrong_actions))
        logits = (
            self.reward_predictor(
                states,
                actions,
                inference=True,
                inference_batch_size=self.config.inference_batch_size,
            )
            .squeeze(1)
            .cpu()
        )
        # equal to sigmoid > 0.5
        predict_labels = (logits > 0).numpy()
        is_right_max = t.argmax(logits).item() == 0
        return {
            "idx": batch_idx,
            "accuracy": accuracy_score(labels, predict_labels),
            "f1": f1_score(labels, predict_labels),
            "is_right_max": is_right_max,
        }

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
        for prefix, dataloader_idx in (("val", 0), ("test", 1)):
            existing_idx = set()
            filtered_outputs = []
            for o in outputs[dataloader_idx]:
                if o["idx"] not in existing_idx:
                    filtered_outputs.append(o)
                    existing_idx.add(o["idx"])
            outputs[dataloader_idx] = filtered_outputs

            average_accuracy = float(
                np.mean([o["accuracy"] for o in outputs[dataloader_idx]])
            )
            average_f1 = float(np.mean([o["f1"] for o in outputs[dataloader_idx]]))
            average_is_right_max_accuracy = float(
                np.mean([o["is_right_max"] for o in outputs[dataloader_idx]])
            )
            self.log(
                f"{prefix}_accuracy", average_accuracy, prog_bar=True, sync_dist=True,
            )
            self.log(f"{prefix}_f1", average_f1, prog_bar=True, sync_dist=True)
            self.log(
                f"{prefix}_is_right_max_accuracy",
                average_is_right_max_accuracy,
                prog_bar=True,
                sync_dist=True,
            )
            if not self.is_distributed or get_rank() == 0:
                print(f"Validation on {prefix} result:")
                print(f"{prefix}_accuracy: {average_accuracy}")
                print(f"{prefix}_f1: {average_f1}")
                print(
                    f"{prefix}_is_right_max_accuracy: {average_is_right_max_accuracy}"
                )

    # def on_test_start(self) -> None:
    #     # print(f"Rank: {get_rank()}, min_logits={self.config.min_logits}")

    def test_step(self, batch: BatchEncoding, _batch_idx, _dataloader_id):
        # print(f"Inferenced action num: {batch[0][-1]}")
        if batch[0][1] is None:
            return None
        return batch[0]

    def test_epoch_end(self, outputs):
        flattened_outputs = [
            ((split, idx), oo)
            for o, split in zip(outputs, (0, 1, 2))
            for idx, oo in enumerate(o)
        ]
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, flattened_outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_every_process(gathered_outputs)
        else:
            self.test_on_every_process(flattened_outputs)

    def test_on_every_process(
        self, outputs: List[Tuple[Tuple[int, int], Tuple[str, List[str], int]]]
    ):
        if not self.is_distributed or get_rank() == 0:
            result = self.test_result
            sorted_outputs = [o[1][:-1] for o in sorted(outputs, key=lambda o: o[0])]
            for id, paths, path_edges in sorted_outputs:
                result[id] = (paths, path_edges)
            with open(
                os.path.join(preprocess_cache_dir, "openbook_qa_sample_result.json"),
                "w",
            ) as file:
                json.dump(result, file)

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

    def create_sample_dataloader(self, split):
        return DataLoader(
            dataset=self.reward_predictor_datasets[split],
            batch_size=self.config.batch_size if split == "train" else 1,
            collate_fn=lambda x: x,
        )

    def create_sample_inference_dataloader(self, split):
        return DataLoader(
            dataset=RewardPredictorBestFirstBeamSearchDataset(
                # [
                #     (d["id"], d["text_question"], ", ".join(d["choices"]),)
                #     for d in getattr(self.dataset, f"original_{split}_data")
                # ][:520]
                # if split == "train"
                # else [
                #     (d["id"], d["text_question"], ", ".join(d["choices"]),)
                #     for d in getattr(self.dataset, f"original_{split}_data")
                # ][:2],
                [
                    (d["id"], d["text_question"], ", ".join(d["choices"]),)
                    for d in getattr(self.dataset, f"original_{split}_data")
                ],
                self.reward_predictor,
                self.dataset.matcher,
                existing_ids=set(self.test_result.keys()),
                max_steps=self.config.max_steps,
                max_depth=self.config.max_depth,
                beam_size=self.config.beam_size,
                return_beam_num=min(self.config.beam_size, self.config.return_beam_num),
                min_logits=self.config.min_logits,
                max_inference_num=self.config.max_inference_num,
                inference_batch_size=self.config.inference_batch_size,
                state_delimiter=self.config.state_delimeter,
                end_of_reasoning=self.config.end_of_reasoning,
            ),
            batch_size=1,
            collate_fn=lambda x: x,
        )
