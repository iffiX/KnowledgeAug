import re
import os
import logging
import torch as t
import pytorch_lightning as pl
from ..utils.config import *
from .commonsense_qa_augment_trainer import CommonsenseQAAugmentTrainer
from .commonsense_qa_sample_trainer import CommonsenseQASampleTrainer
from .openbook_qa_trainer import OpenBookQATrainer
from .openbook_qa_sample_trainer import OpenBookQASampleTrainer
from .openbook_qa_augment_trainer import OpenBookQAAugmentTrainer
from .arc_trainer import ARCTrainer
from .ensemble_trainer import EnsembleTrainer
from .test_distributed_trainer import TestDistributedTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DeepSpeedPlugin

stage_name_to_trainer_map = {
    "commonsense_qa_sample": CommonsenseQASampleTrainer,
    "commonsense_qa_augment": CommonsenseQAAugmentTrainer,
    "openbook_qa": OpenBookQATrainer,
    "openbook_qa_sample": OpenBookQASampleTrainer,
    "openbook_qa_augment": OpenBookQAAugmentTrainer,
    "test_distributed": TestDistributedTrainer,
    "arc": ARCTrainer,
    "ensemble": EnsembleTrainer,
}


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
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}-"
        + stage_trainer.monitor
        + "-{"
        + stage_trainer.monitor
        + ":.3f}",
        save_top_k=1 if getattr(stage_config, "save", False) else 0,
        save_last=getattr(stage_config, "save_last", False),
        monitor=stage_trainer.monitor,
        mode=stage_trainer.monitor_mode,
        verbose=True,
    )
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
        plugins.append(DDPPlugin(find_unused_parameters=True))

    logging.info(f"Precision: {config.precision}")

    trainer = pl.Trainer(
        accelerator="gpu"
        if not (isinstance(config.gpus, int) and config.gpus == 0)
        else None,
        gpus=config.gpus,
        plugins=plugins if len(plugins) > 0 else None,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[t_logger],
        reload_dataloaders_every_epoch=True
        if isinstance(stage_trainer, (OpenBookQASampleTrainer),)
        else False,
        limit_train_batches=getattr(stage_config, "train_steps", None) or 1.0,
        limit_val_batches=getattr(stage_config, "validate_steps", None) or 1.0,
        num_sanity_val_steps=0 if type(stage_trainer) == EnsembleTrainer else 2,
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

        checkpoint = find_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise RuntimeError("Cannot find a valid checkpoint.")
        else:
            logging.info(f"Using checkpoint {checkpoint}")

        # model` must be provided to `trainer.test()` when it hasn't been passed
        # in a previous run.
        # the ckpt_path in test will be ignored in this case.
        # and must perform manual load
        stage_trainer = stage_name_to_checkpoint(stage, checkpoint)

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
            # if isinstance(stage_trainer, OpenBookQAAugmentTrainer):
            #     trainer.fit(stage_trainer)
            # else:
            #     trainer.validate(stage_trainer)
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
