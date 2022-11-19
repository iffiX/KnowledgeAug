import os
import json
import itertools
import warnings
import random
import numpy as np
import torch as t
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Tuple
from torch.utils.data import DataLoader, RandomSampler
from torch.distributed import (
    all_gather_object,
    get_world_size,
    get_rank,
    is_initialized,
)
from transformers import BatchEncoding
from .utils import make_scheduler
from encoder.models.sample.model import RewardPredictor
from encoder.dataset.social_iqa import SocialIQABaseDataset
from encoder.dataset.sample import (
    RewardPredictorSingleChoiceDataset,
    RewardPredictorSingleChoiceDatasetCreatorWithLimitedNodes,
    RewardPredictorSingleChoiceBestFirstBeamSearchDatasetWithLimitedNodes,
)
from encoder.utils.config import SocialIQASingleChoiceSampleTrainConfig, fix_missing
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.adafactor import Adafactor


class SocialIQASingleChoiceSampleTrainer(pl.LightningModule):
    def __init__(
        self,
        config: SocialIQASingleChoiceSampleTrainConfig,
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
        self.dataset = SocialIQABaseDataset(tokenizer=None,)
        self.reward_predictor = RewardPredictor(
            config.base_type,
            pad_by_longest=config.pad_by_longest,
            max_length=config.max_seq_length,
        )
        self.train_reward_predictor_dataset = RewardPredictorSingleChoiceDataset(
            f"social_iqa_train",
            [
                (
                    d["id"],
                    d["context"] + " Context: " + d["question"],
                    d["text_answer"],
                    d["choices"],
                    d["facts"],
                    d["choice_masks"],
                )
                for d in getattr(self.dataset, f"train_data")
                if d["facts"]
            ],
            self.dataset.matcher,
            max_depth=self.config.max_depth,
            negative_samples=self.config.negative_samples,
            negative_shuffle_seed=self.config.negative_shuffle_seed
            if not is_initialized()
            else self.config.negative_shuffle_seed + get_rank(),
            state_delimiter=self.config.state_delimeter,
            end_of_reasoning=self.config.end_of_reasoning,
            creator=RewardPredictorSingleChoiceDatasetCreatorWithLimitedNodes,
        )
        rand = random.Random(42)
        self.validate_indices = rand.choices(
            [
                i
                for i in range(len(self.dataset.validate_data))
                if self.dataset.validate_data[i]["facts"]
            ],
            k=100,
        )
        self.test_result = {}

    @property
    def monitor(self):
        return "retrieval"

    @property
    def monitor_mode(self):
        return "max"

    def export_model(self):
        return self.reward_predictor

    def train_dataloader(self):
        gen = t.Generator()
        gen.manual_seed(42)
        return DataLoader(
            dataset=self.train_reward_predictor_dataset,
            sampler=RandomSampler(self.train_reward_predictor_dataset, generator=gen),
            batch_size=self.config.batch_size,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return self.create_sample_inference_dataloader_for_validate()

    def test_dataloader(self):
        path = os.path.join(preprocess_cache_dir, "social_iqa_sample_result_sc.json")
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
        for _, state, action, wrong_actions in batch:
            states += [state] * (1 + len(wrong_actions))
            actions += [action] + wrong_actions
            labels += [1] + [0] * len(wrong_actions)
        neg_to_pos_ratio = (len(labels) - len(batch)) / len(batch)
        labels = t.Tensor(labels).unsqueeze(1).to(self.device)
        logits = self.reward_predictor(states, actions)
        loss = nn.BCEWithLogitsLoss(
            reduction="mean", pos_weight=t.full([1], neg_to_pos_ratio).to(self.device)
        )(logits, labels)
        return loss

    # noinspection PyTypeChecker
    def validation_step(self, batch, batch_idx):
        _, paths, *__ = batch[0]
        best_length = 0
        for path in paths:
            length = 0
            for fact in self.dataset.validate_data[self.validate_indices[batch_idx]][
                "facts"
            ]:
                if fact in " ".join(path):
                    length += 1
                else:
                    break
            if best_length < length:
                best_length = length

        if self.dataset.validate_data[self.validate_indices[batch_idx]]["facts"]:
            should_add = 1
            if best_length == 0:
                data = self.dataset.validate_data[self.validate_indices[batch_idx]]
                print(f"Question and context: {data['question']} {data['context']}")
                print(f"Answer: {data['text_answer']}")
                print(f"Choices: {data['text_choices']}")
                print(f"Facts: {data['facts']}")
                for idx, path in enumerate(paths):
                    print(f"Path {idx}: {' '.join(path)}")
        else:
            should_add = 0

        return {"idx": batch_idx, "best_length": best_length, "should_add": should_add}

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

        total = np.sum([o["should_add"] for o in filtered_outputs])
        average_retrieval = float(
            np.sum([o["best_length"] for o in filtered_outputs]) / total
        )
        self.log(
            f"retrieval", average_retrieval, prog_bar=True, sync_dist=True,
        )
        if not self.is_distributed or get_rank() == 0:
            print(f"Validation result:")
            print(f"retrieval length: {average_retrieval}")
            print(f"samples with reference facts: {total}")

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
            for id_, paths, path_edges, path_choices in sorted_outputs:
                result[id_] = (paths, path_edges, path_choices)

            with open(
                os.path.join(preprocess_cache_dir, "social_iqa_sample_result_sc.json"),
                "w",
            ) as file:
                json.dump(result, file, indent=2)

            for split_name, split in zip(
                ("train", "validate"),
                (self.dataset.train_data, self.dataset.validate_data,),
            ):
                total, fact_retrieval_count = 0, 0
                for data in split:
                    if data["id"] in result:
                        best_length = 0

                        for path in result[data["id"]][0]:
                            length = 0
                            for fact in data["facts"]:
                                if fact in " ".join(path):
                                    length += 1
                                else:
                                    break
                            if best_length < length:
                                best_length = length

                        if data["facts"]:
                            total += 1
                            fact_retrieval_count += best_length
                            if best_length == 0:
                                print(data["id"])
                print(
                    f"Average retrieval length of {split_name}: "
                    f"{fact_retrieval_count / total}"
                )

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

    def create_sample_inference_dataloader_for_validate(self):
        data = self.dataset.validate_data
        return DataLoader(
            dataset=RewardPredictorSingleChoiceBestFirstBeamSearchDatasetWithLimitedNodes(
                [
                    (
                        data[i]["id"],
                        data[i]["context"] + " Context: " + data[i]["question"],
                        data[i]["choices"],
                        data[i]["choice_masks"],
                    )
                    for i in self.validate_indices
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
                expand_choice_num=self.config.expand_choice_num,
                inference_batch_size=self.config.inference_batch_size,
                state_delimiter=self.config.state_delimeter,
                end_of_reasoning=self.config.end_of_reasoning,
                parallel=not self.is_distributed,
                stop_when_reaching_target_nodes=False,
                prune_low_probability_edges=False,
            ),
            batch_size=1,
            collate_fn=lambda x: x,
        )

    def create_sample_inference_dataloader(self, split):
        return DataLoader(
            dataset=RewardPredictorSingleChoiceBestFirstBeamSearchDatasetWithLimitedNodes(
                [
                    (
                        d["id"],
                        d["context"] + " Context: " + d["question"],
                        d["choices"],
                        d["choice_masks"],
                    )
                    for d in getattr(self.dataset, f"{split}_data")
                    if self.dataset.matcher.allowed_composite_nodes[d["id"]]
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
                expand_choice_num=self.config.expand_choice_num,
                inference_batch_size=self.config.inference_batch_size,
                state_delimiter=self.config.state_delimeter,
                end_of_reasoning=self.config.end_of_reasoning,
                prune_low_probability_edges=False,
            ),
            batch_size=1,
            collate_fn=lambda x: x,
        )
