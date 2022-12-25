import re
import os
import json
import logging
import torch as t
import pytorch_lightning as pl
from ..utils.config import *
from .bespoke_base_trainer import BespokeBaseTrainer
from .social_iqa_augment_trainer import SocialIQAAugmentTrainer
from .social_iqa_sample_trainer import SocialIQASingleChoiceSampleTrainer
from .commonsense_qa2_augment_trainer import CommonsenseQA2AugmentTrainer
from .openbook_qa_sc_sample_trainer import OpenBookQASingleChoiceSampleTrainer
from .openbook_qa_mc_sample_trainer import OpenBookQAMultipleChoiceSampleTrainer
from .openbook_qa_augment_trainer import OpenBookQAAugmentTrainer
from .openbook_qa_bespoke_augment_trainer import OpenBookQABespokeAugmentTrainer
from .qasc_sc_sample_trainer import QASCSingleChoiceSampleTrainer
from .qasc_mc_sample_trainer import QASCMultipleChoiceSampleTrainer
from .qasc_augment_trainer import QASCAugmentTrainer
from .qasc_bespoke_augment_trainer import QASCBespokeAugmentTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

stage_name_to_trainer_map = {
    "social_iqa_sc_sample": SocialIQASingleChoiceSampleTrainer,
    "social_iqa_augment": SocialIQAAugmentTrainer,
    "commonsense_qa2_augment": CommonsenseQA2AugmentTrainer,
    "openbook_qa_sc_sample": OpenBookQASingleChoiceSampleTrainer,
    "openbook_qa_mc_sample": OpenBookQAMultipleChoiceSampleTrainer,
    "openbook_qa_augment": OpenBookQAAugmentTrainer,
    "openbook_qa_bespoke_augment": OpenBookQABespokeAugmentTrainer,
    "qasc_sc_sample": QASCSingleChoiceSampleTrainer,
    "qasc_mc_sample": QASCMultipleChoiceSampleTrainer,
    "qasc_augment": QASCAugmentTrainer,
    "qasc_bespoke_augment": QASCBespokeAugmentTrainer,
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def find_checkpoint(
    checkpoint_path: str, monitor: str = None, monitor_mode: str = None
):
    available_files = []
    if monitor is None or monitor_mode is None:
        logging.info("Finding last available checkpoint")
        for file in os.listdir(checkpoint_path):
            if file.endswith(".ckpt"):
                available_files.append(file)
        sorted_by_time = sorted(
            available_files,
            key=lambda f: os.stat(os.path.join(checkpoint_path, f)).st_mtime,
        )
        if len(sorted_by_time) == 0:
            return None
        checkpoint = sorted_by_time[-1]
    else:
        logging.info(
            f"Finding checkpoint with monitor={monitor}, monitor_mode={monitor_mode}"
        )
        for file in os.listdir(checkpoint_path):
            if re.search(
                f"{monitor}=([+-]?([0-9]*[.])?[0-9]+)", file
            ) is not None and file.endswith(".ckpt"):
                available_files.append(file)
        sorted_by_epoch = sorted(
            available_files,
            key=lambda f: float(
                re.search(f"{monitor}=([+-]?([0-9]*[.])?[0-9]+)", f)[1]
            ),
        )
        if len(sorted_by_epoch) == 0:
            return None
        if monitor_mode == "max":
            checkpoint = sorted_by_epoch[-1]
        else:
            checkpoint = sorted_by_epoch[0]

    return os.path.join(checkpoint_path, checkpoint)


def stage_name_to_trainer(
    stage: str, stage_config, stage_result_path: str, is_distributed: bool
):
    if stage in stage_name_to_trainer_map:
        return stage_name_to_trainer_map[stage](
            stage_config, stage_result_path, is_distributed=is_distributed
        )
    else:
        raise ValueError(f"Unknown stage {stage}.")


def stage_name_to_checkpoint(stage: str, checkpoint_path: str):
    if stage in stage_name_to_trainer_map:
        return stage_name_to_trainer_map[stage].load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
    else:
        raise ValueError(f"Unknown stage {stage}.")


def override_saved_config(trainer, override_attributes):
    print(f"Overriding config for trainer of type {type(trainer)}")
    for k, v in override_attributes.items():
        if hasattr(trainer.config, k):
            print(f"Set {k}={v}, original={getattr(trainer.config, k)}")
            setattr(trainer.config, k, v)
        else:
            print(f"Attribute {k} not found, skipping")


def _train(
    config,
    stage_config,
    stage_trainer,
    is_distributed: bool,
    checkpoint_path: str,
    log_path: str,
):
    # create directories, or reuse
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    save_config(config, os.path.join(config.working_directory, "config.json"))
    save_top_k = int(getattr(stage_config, "save", False))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}-"
        + stage_trainer.monitor
        + "-{"
        + stage_trainer.monitor
        + ":.3f}",
        save_top_k=save_top_k,
        save_last=getattr(stage_config, "save_last", False),
        monitor=stage_trainer.monitor,
        mode=stage_trainer.monitor_mode,
        verbose=True,
    )
    stage_trainer.current_mode = "train"
    early_stopping = EarlyStopping(
        monitor=stage_trainer.monitor,
        mode=stage_trainer.monitor_mode,
        patience=config.early_stopping_patience,
        verbose=True,
    )
    t_logger = TensorBoardLogger(log_path)

    checkpoint = None
    if getattr(stage_config, "load", False):
        checkpoint = find_checkpoint(
            checkpoint_path, stage_trainer.monitor, stage_trainer.monitor_mode
        )
        if checkpoint is None:
            logging.info("Failed to find a valid checkpoint, using original weights.")
        else:
            logging.info(f"Using checkpoint {checkpoint}")
    else:
        logging.info("Not loading, using original weights.")

    plugins = []

    if config.deepspeed:
        deepspeed_configs = config.deepspeed_configs or {
            "cpu_offload": True,
            "allgather_bucket_size": 2e8,
            "reduce_bucket_size": 2e8,
        }
        plugins.append(DeepSpeedPlugin(**deepspeed_configs))
    elif is_distributed:
        plugins.append(DDPPlugin(find_unused_parameters=False))

    logging.info(f"Precision: {config.precision}")

    trainer = pl.Trainer(
        accelerator="gpu"
        if not (isinstance(config.gpus, int) and config.gpus == 0)
        else None,
        gpus=config.gpus,
        plugins=plugins if len(plugins) > 0 else None,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[t_logger],
        reload_dataloaders_every_epoch=False,
        limit_train_batches=getattr(stage_config, "train_steps", None) or 1.0,
        limit_val_batches=getattr(stage_config, "validate_steps", None) or 1.0,
        max_epochs=getattr(stage_config, "epochs", 1),
        # # For iterable datasets, to validate after each epoch,
        # # set check interval equal to number of training steps.
        # val_check_interval=stage_config.train_steps,
        accumulate_grad_batches=getattr(stage_config, "accumulate_grad_batches", 1),
        resume_from_checkpoint=checkpoint,
        # deterministic=True,
        precision=config.precision,
    )
    trainer.stage_mode = "train"
    trainer.fit(stage_trainer)


def run(config: Config, stage_index: int, mode: str = "train"):
    # t.multiprocessing.set_start_method("spawn", force=True)
    # execute stages
    stage = config.stages[stage_index]

    is_distributed = (isinstance(config.gpus, list) and len(config.gpus) > 1) or (
        isinstance(config.gpus, int) and config.gpus > 1
    )
    stage_config = config.configs[stage_index]
    seed_everything(getattr(stage_config, "seed", 42), workers=True)
    if getattr(stage_config, "load_worker_num", 0) > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    checkpoint_path = os.path.join(
        config.working_directory, str(stage_index), "checkpoint"
    )
    log_path = os.path.join(config.working_directory, str(stage_index), "log")
    stage_result_path = os.path.join(
        config.working_directory, str(stage_index), "result"
    )

    if mode == "train":
        stage_trainer = stage_name_to_trainer(
            stage, stage_config, stage_result_path, is_distributed
        )

        logging.info("Training.")
        _train(
            config=config,
            stage_config=stage_config,
            stage_trainer=stage_trainer,
            is_distributed=is_distributed,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
        )
    elif mode in ("validate", "test"):
        logging.info("Validating." if mode == "validate" else "Testing")

        if issubclass(stage_name_to_trainer_map[stage], BespokeBaseTrainer):
            stage_trainer = stage_name_to_trainer(
                stage, stage_config, stage_result_path, is_distributed
            )
        else:
            checkpoint = find_checkpoint(checkpoint_path)
            if checkpoint is None:
                raise RuntimeError("Cannot find a valid checkpoint.")
            else:
                logging.info(f"Using checkpoint {checkpoint}")
            stage_trainer = stage_name_to_checkpoint(stage, checkpoint)

        stage_trainer.current_mode = mode

        is_distributed = (isinstance(config.gpus, list) and len(config.gpus) > 1) or (
            isinstance(config.gpus, int) and config.gpus > 1
        )

        stage_trainer.is_distributed = is_distributed
        if (
            config.override_saved_config is not None
            and str(stage_index) in config.override_saved_config
        ):
            override_saved_config(
                stage_trainer, config.override_saved_config[str(stage_index)]
            )

        trainer = pl.Trainer(
            accelerator="gpu"
            if not (isinstance(config.gpus, int) and config.gpus == 0)
            else None,
            gpus=config.gpus,
            plugins=[DDPPlugin(find_unused_parameters=True)]
            if is_distributed
            else None,
            # deterministic=True,
        )
        trainer.stage_mode = mode
        if mode == "validate":
            trainer.validate(stage_trainer)
        else:
            trainer.test(stage_trainer)
    elif mode == "evaluate_model":
        logging.info("Evaluating model.")

        checkpoint = find_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise RuntimeError("Cannot find a valid checkpoint.")
        else:
            logging.info(f"Using checkpoint {checkpoint}")

        stage_trainer = stage_name_to_checkpoint(stage, checkpoint)
        stage_trainer.current_mode = mode

        is_distributed = (isinstance(config.gpus, list) and len(config.gpus) > 1) or (
            isinstance(config.gpus, int) and config.gpus > 1
        )

        stage_trainer.is_distributed = is_distributed
        if (
            config.override_saved_config is not None
            and str(stage_index) in config.override_saved_config
        ):
            override_saved_config(
                stage_trainer, config.override_saved_config[str(stage_index)]
            )
        while True:
            sentence = input("Input: ")
            batch = stage_trainer.tokenizer(sentence, return_tensors="pt",)
            if "t5-" in stage_trainer.config.base_type:
                tokens = stage_trainer.model.generate(
                    input_ids=batch["input_ids"].to(stage_trainer.real_device),
                    attention_mask=batch["attention_mask"].to(
                        stage_trainer.real_device
                    ),
                    max_length=stage_trainer.config.generate_length,
                    early_stopping=True,
                )
                out = [
                    stage_trainer.tokenizer.decode(tokens[i])
                    for i in range(tokens.shape[0])
                ]
            else:
                for predictor in stage_trainer.model.choice_predictors.items():
                    predictor[1].choice_num = 1
                out = stage_trainer.model.predict(
                    input_ids=batch["input_ids"].to(stage_trainer.real_device),
                    attention_mask=batch["attention_mask"].to(
                        stage_trainer.real_device
                    ),
                    token_type_ids=batch["token_type_ids"].to(
                        stage_trainer.real_device
                    ),
                )
                out = t.sigmoid(out)
            print("Output:")
            print(out)
    else:
        raise ValueError(f"Unknown mode {mode}")


def export_model(config: Config, stage_index: int, path: str):
    logging.info("Exporting model state dict.")

    # execute stages
    stage = config.stages[stage_index]
    checkpoint_path = os.path.join(
        config.working_directory, str(stage_index), "checkpoint"
    )

    checkpoint = find_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise RuntimeError("Cannot find a valid checkpoint.")
    else:
        logging.info(f"Using checkpoint {checkpoint}")

    stage_trainer = stage_name_to_checkpoint(stage, checkpoint)
    state_dict = stage_trainer.export_model().state_dict()
    t.save({k: v.cpu() for k, v in state_dict.items()}, path)


def export_bert_input(config: Config, stage_index: int, path: str):
    logging.info("Exporting BERT like input.")

    # execute stages
    stage = config.stages[stage_index]
    stage_config = config.configs[stage_index]

    stage_trainer = stage_name_to_trainer(stage, stage_config, path, False)
    dataset = stage_trainer.dataset
    if dataset.output_mode != "splitted":
        logging.info("Note: Config is not for BERT like models, corrected.")
        dataset.output_mode = "splitted"

    def generate(split):
        results = []
        if split == "train":
            length = len(dataset.train_data)
        elif split == "validate":
            length = len(dataset.validate_data)
        else:
            length = len(dataset.test_data)
        for i in range(length):
            generated = dataset.generator(i, split)
            results.append(
                {
                    "input": generated["bert_input"],
                    "label": generated["bert_label"],
                    "id": generated["id"],
                }
            )
        return results

    if hasattr(dataset, "train_data"):
        with open(os.path.join(path, "train_for_bert.json"), "w") as file:
            json.dump(generate("train"), file, indent=2)

    if hasattr(dataset, "validate_data"):
        with open(os.path.join(path, "validate_for_bert.json"), "w") as file:
            json.dump(generate("validate"), file, indent=2)

    if hasattr(dataset, "test_data"):
        with open(os.path.join(path, "test_for_bert.json"), "w") as file:
            json.dump(generate("test"), file, indent=2)
    logging.info("Export finished")


def export_t5_input(config: Config, stage_index: int, path: str):
    logging.info("Exporting T5 input.")

    # execute stages
    stage = config.stages[stage_index]
    stage_config = config.configs[stage_index]

    stage_trainer = stage_name_to_trainer(stage, stage_config, path, False)
    dataset = stage_trainer.dataset
    if dataset.output_mode != "single":
        logging.info("Note: Config is not using T5, corrected.")
        dataset.output_mode = "single"

    def generate(split):
        results = []
        if split == "train":
            length = len(dataset.train_data)
        elif split == "validate":
            length = len(dataset.validate_data)
        else:
            length = len(dataset.test_data)
        for i in range(length):
            generated = dataset.generator(i, split)
            results.append(
                {
                    "input": generated["t5_input"],
                    "answer": generated["t5_answer"],
                    "label": generated["t5_label"],
                    "id": generated["id"],
                }
            )
        return results

    if hasattr(dataset, "train_data"):
        with open(os.path.join(path, "train_for_t5.json"), "w") as file:
            json.dump(generate("train"), file, indent=2)

    if hasattr(dataset, "validate_data"):
        with open(os.path.join(path, "validate_for_t5.json"), "w") as file:
            json.dump(generate("validate"), file, indent=2)

    if hasattr(dataset, "test_data"):
        with open(os.path.join(path, "test_for_t5.json"), "w") as file:
            json.dump(generate("test"), file, indent=2)
    logging.info("Export finished")
