import os
import json
import itertools
import warnings
import numpy as np
import torch as t
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from machin.frame.algorithms.gail import GAIL
from .utils import (
    make_scheduler,
    UncheckedExpertTransition,
    GAILExpert,
    GAILDataset,
    GAILActorPretrainDataset,
    set_worker_sharing_strategy,
    collate_function_dict_to_batch_encoding,
)
from encoder.model.rl.model import (
    MStr,
    PPO,
    TRPO,
    Embedder,
    Actor,
    Critic,
    Discriminator,
)
from encoder.dataset.openbook_qa import OpenBookQADataset
from encoder.utils.file import JSONCache
from encoder.utils.config import OpenBookQAGailTrainConfig, fix_missing
from encoder.utils.settings import preprocess_cache_dir
from encoder.utils.adafactor import Adafactor


class OpenBookQAGailTrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQAGailTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        if is_distributed:
            raise ValueError("Distributed training not supported")
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
        self.embedder = Embedder(config.embedder_base_type, pad_by_longest=True)
        self.actor = Actor(
            config.base_type, self.embedder.output_dim, max_length=config.max_seq_length
        )
        self.critic = Critic(config.base_type, max_length=config.max_seq_length)
        self.discriminator = Discriminator(
            config.base_type, max_length=config.max_seq_length
        )
        self.trpo = TRPO(
            self.actor,
            self.critic,
            getattr(t.optim, self.config.optimizer_class),
            nn.MSELoss(reduction="sum"),
            actor_learning_rate=config.rl_actor_learning_rate,
            critic_learning_rate=config.rl_critic_learning_rate,
            kl_max_delta=0.2,
            batch_size=3,
            value_weight=1,
            replay_device="cpu",
            replay_size=config.replay_size,
        )
        self.gail = GAIL(
            self.discriminator,
            self.trpo,
            t.optim.AdamW,
            batch_size=2,
            discriminator_learning_rate=config.rl_discriminator_learning_rate,
            expert_replay_device="cpu",
            expert_replay_size=config.replay_size,
        )
        self.trpo._backward = self.manual_backward
        self.gail._backward = self.manual_backward
        self.automatic_optimization = False

        self.actor_pretrain_dataset = GAILActorPretrainDataset(
            [
                (d["text_question"], d["text_answer"], d["original_facts"])
                for d in self.dataset.original_train_data
            ],
            self.dataset.matcher,
            self.embedder,
            self.config.state_delimeter,
        )

        with JSONCache(
            os.path.join(preprocess_cache_dir, "openbook_qa_expert.json"),
            self.generate_expert_transitions,
        ) as cache:
            for transitions in cache.data:
                self.gail.store_expert_episode(
                    [
                        UncheckedExpertTransition(
                            state={"state": MStr(trans["state"]["state"])},
                            action={
                                "action_string": MStr(trans["action"]["action_string"])
                            },
                        )
                        for trans in transitions
                    ]
                )

    @property
    def monitor(self):
        return "test_recall"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        if self.should_pretrain():
            return DataLoader(
                dataset=self.actor_pretrain_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.actor_pretrain_batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                worker_init_fn=set_worker_sharing_strategy,
            )
        else:
            return self.create_gail_dataloader("train")

    def val_dataloader(self):
        return [
            self.create_gail_dataloader("validate"),
            self.create_gail_dataloader("test"),
        ]

    def test_dataloader(self):
        return [
            self.create_gail_dataloader("train"),
            self.create_gail_dataloader("validate"),
            self.create_gail_dataloader("test"),
        ]

    def on_train_epoch_start(self):
        self.actor.clear_cache()
        if self.should_pretrain():
            self.actor.is_cache_enabled = False
        else:
            self.actor.is_cache_enabled = True

    def on_train_epoch_end(self):
        self.actor.clear_cache()
        self.actor.is_cache_enabled = False

    # noinspection PyTypeChecker
    def training_step(self, batch, batch_idx):
        if self.should_pretrain():
            # loss = nn.MSELoss(reduction="mean")(
            #     self.actor(batch["state"], sample=False),
            #     batch["action_embedding"].to(self.device),
            # )
            loss = -t.sum(self.actor(batch["state"])[1])
            actor_optim = self.optimizers()[0]
            actor_optim.zero_grad()
            self.manual_backward(loss)
            actor_optim.step()
            self.log(
                "p_loss", loss.item() / len(batch["state"]), prog_bar=True, on_step=True
            )
        else:
            losses = []
            for sample in batch:
                self.gail.store_episode(sample["transitions"])
                losses.append(self.gail.update(concatenate_samples=False))
            self.actor.clear_cache()
            self.log(
                "act_loss", np.mean([l[0] for l in losses]), prog_bar=True, on_step=True
            )
            self.log(
                "value_loss",
                np.mean([l[1] for l in losses]),
                prog_bar=True,
                on_step=True,
            )
            self.log(
                "discrim_loss",
                np.mean([l[2] for l in losses]),
                prog_bar=True,
                on_step=True,
            )

    # noinspection PyTypeChecker
    def validation_step(self, batch, _batch_idx, _dataloader_idx):
        recall = 0
        action_strings = [
            transition["action"]["action_string"]
            for transition in batch[0]["transitions"]
        ]
        for intermediate_node in batch[0]["intermediate_nodes"]:
            for action_string in action_strings:
                if intermediate_node in action_string:
                    recall += 1
                    break
        return recall / len(batch[0]["intermediate_nodes"])

    def validation_epoch_end(self, outputs):
        for prefix, dataloader_idx in (("val", 0), ("test", 1)):
            average_recall = float(np.mean(outputs[dataloader_idx]))
            self.log(f"{prefix}_recall", average_recall, prog_bar=True, sync_dist=True)
            print(f"Validation on {prefix} result:")
            print(f"{prefix}_recall: {average_recall}")

    def test_step(self, batch: BatchEncoding, _batch_idx):
        action_strings = [
            transition["action"]["action_string"]
            for transition in batch[0]["transitions"]
        ]
        return action_strings

    def test_epoch_end(self, outputs):
        result = {}
        for sample, action_strings in zip(
            itertools.chain(
                self.dataset.original_train_data,
                self.dataset.original_validate_data,
                self.dataset.original_test_data,
            ),
            itertools.chain(*outputs),
        ):
            result[sample["id"]] = action_strings
        with open(
            os.path.join(preprocess_cache_dir, "openbook_qa_rl_result.json")
        ) as file:
            json.dump(result, file)

    def configure_optimizers(self):
        return (
            getattr(t.optim, self.config.optimizer_class)(
                self.actor.all_parameters(), lr=self.config.pretrain_learning_rate
            ),
            self.trpo.actor_optim,
            self.trpo.critic_optim,
            self.gail.discriminator_optim,
        )

    def should_pretrain(self):
        return self.current_epoch < self.config.pretrain_epochs

    def generate_expert_transitions(self):
        episodes = []
        for split in ("train", "validate", "test"):
            print(f"Processing expert episodes of split {split}")
            expert = GAILExpert(
                [
                    (d["text_question"], d["text_answer"], d["original_facts"])
                    for d in getattr(self.dataset, f"original_{split}_data")
                ],
                self.dataset.matcher,
                self.config.state_delimeter,
            )
            episodes += expert.get_transitions()
        return episodes

    def create_gail_dataloader(self, split):
        return DataLoader(
            dataset=GAILDataset(
                [
                    (d["text_question"], d["text_answer"], d["original_facts"])
                    for d in getattr(self.dataset, f"original_{split}_data")
                ],
                self.dataset.matcher,
                self.actor,
                self.embedder,
                self.config.max_steps,
                self.config.state_delimeter,
            ),
            batch_size=self.config.batch_size if split == "train" else 1,
            collate_fn=lambda x: x,
        )
