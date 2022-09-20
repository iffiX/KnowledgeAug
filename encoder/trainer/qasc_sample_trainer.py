import os
import copy
import json
import itertools
import warnings
import numpy as np
import torch as t
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
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
from encoder.dataset.qasc import QASCBaseDataset
from encoder.dataset.sample import (
    RewardPredictorDatasetCreatorWithFilter,
    QASCRewardPredictorDataset,
    QASCRewardPredictorBestFirstBeamSearchDataset,
)
from encoder.utils.config import QASCSampleTrainConfig, fix_missing
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.adafactor import Adafactor


class QASCSampleTrainer(pl.LightningModule):
    def __init__(
        self,
        config: QASCSampleTrainConfig,
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
        self.dataset = QASCBaseDataset(tokenizer=None)
        self.reward_predictor = RewardPredictor(
            config.base_type,
            pad_by_longest=config.pad_by_longest,
            max_length=config.max_seq_length,
        )

        matcher = copy.deepcopy(self.dataset.matcher)
        self.dataset.matcher.add_qasc_facts(train_and_validate=False)
        matcher.add_qasc_facts(train_and_validate=True)

        # # Estimate filter bound
        # filter_bound_sum = 0
        # filter_bound_sum_l2 = 0
        # filter_bound_num = 0
        # filter_bound_num_l2 = 0
        # filter_bound_range = [1, 0]
        # filter_bound_range_l2 = [1, 0]
        # for data in tqdm(self.dataset.train_data):
        #     # first level
        #     _, target_nodes = self.dataset.matcher.match_source_and_target_nodes(
        #         "", data["text_question"]
        #     )
        #     nodes = self.dataset.matcher.matcher.kb.find_nodes(data["facts"])
        #     filter_bound_num += 1
        #     score = self.dataset.matcher.matcher.compute_f_beta_score(
        #         nodes[0], target_nodes, beta=2
        #     )
        #     filter_bound_sum += score
        #     filter_bound_range = [
        #         min(filter_bound_range[0], score),
        #         max(filter_bound_range[1], score),
        #     ]
        #
        #     # next level
        #     (_, target_nodes,) = self.dataset.matcher.match_source_and_target_nodes(
        #         "", self.dataset.matcher.matcher.kb.nodes[nodes[0]].lower()
        #     )
        #     filter_bound_num_l2 += 1
        #     score = self.dataset.matcher.matcher.compute_f_beta_score(
        #         nodes[1], target_nodes, beta=2
        #     )
        #     filter_bound_sum_l2 += score
        #     filter_bound_range_l2 = [
        #         min(filter_bound_range_l2[0], score),
        #         max(filter_bound_range_l2[1], score),
        #     ]
        #
        # print(f"F-Beta:{filter_bound_sum / filter_bound_num}")
        # print(f"F-Beta: min={filter_bound_range[0]}, max={filter_bound_range[1]}")
        # print(f"F-Beta-l2:{filter_bound_sum_l2 / filter_bound_num_l2}")
        # print(
        #     f"F-Beta-l2: min={filter_bound_range_l2[0]}, max={filter_bound_range_l2[1]}"
        # )

        self.reward_predictor_datasets = {
            split: QASCRewardPredictorDataset(
                f"qasc_{split}",
                [
                    (
                        d["id"],
                        d["text_question"],
                        d["text_choices"],
                        d["text_answer"],
                        d["original_facts"],
                    )
                    for d in getattr(self.dataset, f"{split}_data")
                ],
                matcher,
                limit_size=limit_size,
                limit_neg_transition_num=4000,
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
            for split, limit_size in (("train", None), ("validate", 100))
        }
        self.test_result = {}

    @property
    def monitor(self):
        return "is_right_max_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    def export_model(self):
        return self.reward_predictor

    def train_dataloader(self):
        return self.create_sample_dataloader("train")

    def val_dataloader(self):
        return self.create_sample_dataloader("validate")

    def test_dataloader(self):
        path = os.path.join(preprocess_cache_dir, "qasc_sample_result.json")
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
    def validation_step(self, batch, batch_idx):
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
        existing_idx = set()
        filtered_outputs = []
        for o in outputs:
            if o["idx"] not in existing_idx:
                filtered_outputs.append(o)
                existing_idx.add(o["idx"])

        average_accuracy = float(np.mean([o["accuracy"] for o in filtered_outputs]))
        average_f1 = float(np.mean([o["f1"] for o in filtered_outputs]))
        average_is_right_max_accuracy = float(
            np.mean([o["is_right_max"] for o in filtered_outputs])
        )
        self.log(
            f"accuracy", average_accuracy, prog_bar=True, sync_dist=True,
        )
        self.log(f"f1", average_f1, prog_bar=True, sync_dist=True)
        self.log(
            f"is_right_max_accuracy",
            average_is_right_max_accuracy,
            prog_bar=True,
            sync_dist=True,
        )
        if not self.is_distributed or get_rank() == 0:
            print(f"Validation result:")
            print(f"accuracy: {average_accuracy}")
            print(f"f1: {average_f1}")
            print(f"is_right_max_accuracy: {average_is_right_max_accuracy}")

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

            for split_name, split in zip(
                ("train", "validate"),
                (self.dataset.train_data, self.dataset.validate_data),
            ):
                total, count = 0, 0
                for data in split:
                    if data["id"] in result:
                        total += 1
                        for fact in data["original_facts"]:
                            for path in result[data["id"]][0]:
                                if fact in " ".join(path):
                                    count += 1
                                    break
                print(f"Average retrieval length of {split_name}: {count / total}")
            with open(
                os.path.join(preprocess_cache_dir, "qasc_sample_result.json"), "w",
            ) as file:
                json.dump(result, file, indent=2)

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
            dataset=QASCRewardPredictorBestFirstBeamSearchDataset(
                [
                    (d["id"], d["text_question"], ", ".join(d["choices"]),)
                    for d in getattr(self.dataset, f"{split}_data")
                ],
                # [
                #     (d["id"], d["text_question"], ", ".join(d["choices"]),)
                #     for d in getattr(self.dataset, f"{split}_data")
                # ][:1]
                # if split in ("train", "test")
                # else [
                #     (d["id"], d["text_question"], ", ".join(d["choices"]),)
                #     for d in getattr(self.dataset, f"{split}_data")
                # ][:20],
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
                stop_when_reaching_target_nodes=False,
            ),
            batch_size=1,
            collate_fn=lambda x: x,
        )
