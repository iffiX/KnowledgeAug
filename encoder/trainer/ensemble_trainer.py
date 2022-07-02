import warnings
import torch as t
import pytorch_lightning as pl
from copy import deepcopy
from collections import Counter
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from .utils import collate_and_filter_outputs
from .commonsense_qa_search_trainer import CommonsenseQASearchTrainer
from .openbook_qa_trainer import OpenBookQATrainer
from encoder.dataset.base import SizeOnePlaceholderDataset
from encoder.utils.config import EnsembleTrainConfig


class EnsembleTrainer(pl.LightningModule):
    def __init__(
        self, config: EnsembleTrainConfig, stage_result_path="./", **_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        self.config = config
        self.stage_result_path = stage_result_path

        if len(config.checkpoints) == 0:
            raise ValueError("Checkpoint must be non-empty")

        self.trainers = []
        if (
            len(
                {
                    len(config.matcher_configs_list),
                    len(config.matcher_modes_list),
                    len(config.matcher_seeds_list),
                }
            )
            != 1
        ):
            raise ValueError(
                "Matcher config list length, match mode list length, and seed list "
                "length does not match"
            )

        for chk, matcher_configs, matcher_modes, matcher_seeds in zip(
            config.checkpoints,
            config.matcher_configs_list,
            config.matcher_modes_list,
            config.matcher_seeds_list,
        ):
            trainer = self.stage_name_to_checkpoint(config.task_trainer_stage, chk)
            model = trainer.model
            sub_trainers = []

            if len({len(matcher_configs), len(matcher_modes), len(matcher_seeds)}) != 1:
                raise ValueError(
                    f"Matcher config length, match mode length, and seed length "
                    f"of checkpoint {chk} does not match"
                )

            for _ in range(len(matcher_configs)):
                # Share the same model to reduce memory usage
                sub_trainer = deepcopy(trainer, memo={id(model): model})
                sub_trainers.append(sub_trainer)

            for trainer, m_mode, m_seed, m_config in zip(
                sub_trainers, matcher_modes, matcher_seeds, matcher_configs,
            ):
                setattr(trainer.dataset, "matcher_mode", m_mode)
                setattr(trainer.dataset, "matcher_seed", m_seed)
                setattr(trainer.dataset, "matcher_config", m_config)

            self.trainers += sub_trainers

        self.on_gpu_trainer = -1

    @property
    def monitor(self):
        return self.trainers[0].monitor

    @property
    def monitor_mode(self):
        return self.trainers[0].monitor_mode

    def train_dataloader(self):
        return DataLoader(dataset=SizeOnePlaceholderDataset())

    def val_dataloader(self):
        return [trainer.val_dataloader() for trainer in self.trainers]

    def test_dataloader(self):
        return [trainer.test_dataloader() for trainer in self.trainers]

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        return None

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, batch_idx, dataloader_idx):
        self.move_trainers(dataloader_idx)
        self.trainers[dataloader_idx].real_device = self.device
        result = self.trainers[dataloader_idx].validation_step(batch, batch_idx)
        return {
            "batch": result["batch"],
            "tokens": result["tokens"],
        }

    def validation_epoch_end(self, outputs):
        batch, tokens = self.select_answer(outputs)
        metrics = self.trainers[0].dataset.validate_tokens(batch, tokens)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        print("Validation result:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    def test_step(self, batch: BatchEncoding, batch_idx, dataloader_idx):
        self.move_trainers(dataloader_idx)
        self.trainers[dataloader_idx].real_device = self.device
        result = self.trainers[dataloader_idx].test_step(batch, batch_idx)
        return {
            "batch": result["batch"],
            "tokens": result["tokens"],
        }

    def test_epoch_end(self, outputs):
        batch, tokens = self.select_answer(outputs)
        self.trainers[0].dataset.generate_test_results(tokens, self.stage_result_path)

    def configure_optimizers(self):
        return None

    def select_answer(self, outputs):
        batch = None
        answers_dict = {}
        tokenizer = self.trainers[0].retriever_tokenizer
        all_batches = {}
        for i in range(len(self.trainers)):
            batch, tokens = collate_and_filter_outputs(outputs[i])
            all_batches[i] = batch
            answers_dict[i] = {
                batch["id"][j]: (
                    tokenizer.decode(tokens[j], skip_special_tokens=True),
                    tokens[j],
                )
                for j in range(len(batch["id"]))
            }
        # Find answers according to the order of the last batch
        selected_tokens = []
        any_correct = 0
        for idx, sample_id in enumerate(batch["id"]):
            answers = [
                (i, answers_dict[i][sample_id][0]) for i in range(len(self.trainers))
            ]

            sentences = [
                tokenizer.decode(
                    all_batches[i]["sentence"][idx], skip_special_tokens=True
                )
                for i in range(len(self.trainers))
            ]
            ref_answer_tensor = batch["answer"][idx]
            ref_answer_tensor.masked_fill_(
                ref_answer_tensor == -100, tokenizer.pad_token_id
            )
            ref_answer = tokenizer.decode(ref_answer_tensor, skip_special_tokens=True)
            aa = list(Counter([a[1] for a in answers]))
            any_correct += any(a == ref_answer for a in aa) and len(aa) <= 2
            print("sentence:\n")
            for sentence in sentences:
                print(f"[{sentence}]")
            print(f"answers: [{answers}] \n" f"ref_answer: [{ref_answer}]")

            answer = Counter([a[1] for a in answers]).most_common(1)[0][0]
            selected_trainer_index = [a[0] for a in answers if a[1] == answer][0]
            selected_tokens.append(
                answers_dict[selected_trainer_index][sample_id][1].unsqueeze(0)
            )

        # Any batch can be used to check the result, they are sampled in the same order
        print(f"Any correct: {any_correct / len(batch['id'])}")
        return batch, t.cat(selected_tokens, dim=0)

    def move_trainers(self, trainer_index):
        if self.on_gpu_trainer == -1:
            print(f"Moving trainer {trainer_index} to GPU")
            self.trainers[trainer_index].to(self.device)
            self.on_gpu_trainer = trainer_index
        elif trainer_index != self.on_gpu_trainer:
            print(f"Moving trainer {self.on_gpu_trainer} to CPU")
            self.trainers[self.on_gpu_trainer].to("cpu")
            print(f"Moving trainer {trainer_index} to GPU")
            self.trainers[trainer_index].to(self.device)
            self.on_gpu_trainer = trainer_index

    @classmethod
    def stage_name_to_checkpoint(cls, stage: str, checkpoint_path: str):
        stage_name_to_trainer_map = {
            "commonsense_qa": CommonsenseQASearchTrainer,
            "openbook_qa": OpenBookQATrainer,
        }
        if stage in stage_name_to_trainer_map:
            return stage_name_to_trainer_map[stage].load_from_checkpoint(
                checkpoint_path, "cpu"
            )
        else:
            raise ValueError(f"Unknown stage {stage}.")
