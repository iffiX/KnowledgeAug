import json
from pydantic import BaseModel
from pprint import pprint
from typing import *


class CommonsenseQATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    previous_train_checkpoint_path: Optional[str] = None
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "t5-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class OpenBookQATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "t5-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class OpenBookQASampleTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 1
    accumulate_grad_batches: int = 1

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-base"
    pad_by_longest: bool = True
    max_seq_length: Union[int, None] = None
    inference_batch_size: int = 256

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    match_closest_when_no_equal: bool = True

    max_steps: int = 2
    max_depth: int = 2
    beam_size: int = 10
    return_beam_num: int = 5
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    negative_samples: int = 31
    negative_shuffle_seed: int = 42


class OpenBookQAAugmentTrainConfig(BaseModel):
    load: bool = False
    seed: int = 42
    save: bool = True
    save_last: bool = False
    epochs: int = 20
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 16
    accumulate_grad_batches: int = 1

    optimizer_class: str = "AdamW"
    learning_rate: float = 5e-6
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 256
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class QASCSampleTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 1
    accumulate_grad_batches: int = 1

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-base"
    pad_by_longest: bool = True
    max_seq_length: Union[int, None] = None
    inference_batch_size: int = 256

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    match_closest_when_no_equal: bool = True

    max_steps: int = 3
    max_depth: int = 2
    beam_size: int = 5
    return_beam_num: int = 5
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    negative_samples: int = 31
    negative_shuffle_seed: int = 42


class QASCAugmentTrainConfig(BaseModel):
    load: bool = False
    seed: int = 42
    save: bool = True
    save_last: bool = False
    epochs: int = 20
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 16
    accumulate_grad_batches: int = 1

    optimizer_class: str = "AdamW"
    learning_rate: float = 5e-6
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 256
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class CommonsenseQASampleTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 1
    accumulate_grad_batches: int = 1

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-base"
    initial_weight_path: str = ""
    pad_by_longest: bool = True
    max_seq_length: Union[int, None] = None
    inference_batch_size: int = 128

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_steps: int = 2
    max_depth: int = 2
    beam_size: int = 10
    return_beam_num: int = 5
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    negative_samples: int = 31
    negative_shuffle_seed: int = 42


class CommonsenseQAAugmentTrainConfig(BaseModel):
    load: bool = False
    seed: int = 42
    save: bool = True
    save_last: bool = False
    epochs: int = 20
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 16
    accumulate_grad_batches: int = 1

    optimizer_class: str = "AdamW"
    learning_rate: float = 5e-6
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 256
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class CommonsenseQA2AugmentTrainConfig(BaseModel):
    load: bool = False
    seed: int = 42
    save: bool = True
    save_last: bool = False
    epochs: int = 20
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 16
    accumulate_grad_batches: int = 1

    optimizer_class: str = "AdamW"
    learning_rate: float = 5e-6
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 256
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class ARCTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    previous_train_checkpoint_path: Optional[str] = None
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "microsoft/deberta-v3-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 128
    generate_length: int = 10
    device_map: Optional[Dict[int, List[int]]] = None
    pipe_chunks: Optional[int] = 8
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None


class EnsembleTrainConfig(BaseModel):
    task_trainer_stage: str
    checkpoints: List[str]
    matcher_modes_list: List[List[str]]
    matcher_seeds_list: List[List[int]]
    matcher_configs_list: List[List[dict]]


class TestDistributedTrainConfig(BaseModel):
    train_batch_size: int = 2
    validate_batch_size: int = 2
    test_batch_size: int = 2
    train_dataset_size: int = 10
    validate_dataset_size: int = 10
    test_dataset_size: int = 10

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class Config(BaseModel):
    # Cuda ids of GPUs
    gpus: Optional[Union[int, List[int]]] = [0]
    precision: Optional[Union[int, str]] = 32
    # Only available for validate/test
    override_saved_config: Optional[Dict[str, Dict[str, Any]]] = None
    deepspeed: bool = False
    deepspeed_configs: Optional[dict] = None
    # Maximum validation epochs allowed before stopping
    # when monitored metric is not decreasing
    early_stopping_patience: int = 100

    # Path to the working directory
    # sub-stages will be created as 0, 1, ... subdirectories
    working_directory: str = "./train"

    # example: ["kb_encoder", "qa"]
    # config in configs must match items in stages
    stages: List[str] = []
    configs: List[
        Union[
            CommonsenseQATrainConfig,
            CommonsenseQASampleTrainConfig,
            CommonsenseQAAugmentTrainConfig,
            CommonsenseQA2AugmentTrainConfig,
            OpenBookQATrainConfig,
            OpenBookQASampleTrainConfig,
            OpenBookQAAugmentTrainConfig,
            QASCSampleTrainConfig,
            QASCAugmentTrainConfig,
            EnsembleTrainConfig,
        ]
    ] = []


def stage_name_to_config(name: str, config_dict: dict = None):
    stage_name_to_config_map = {
        "commonsense_qa": CommonsenseQATrainConfig,
        "commonsense_qa_sample": CommonsenseQASampleTrainConfig,
        "commonsense_qa_augment": CommonsenseQAAugmentTrainConfig,
        "commonsense_qa2_augment": CommonsenseQA2AugmentTrainConfig,
        "openbook_qa": OpenBookQATrainConfig,
        "openbook_qa_sample": OpenBookQASampleTrainConfig,
        "openbook_qa_augment": OpenBookQAAugmentTrainConfig,
        "qasc_sample": QASCSampleTrainConfig,
        "qasc_augment": QASCAugmentTrainConfig,
        "ensemble": EnsembleTrainConfig,
        "test_distributed": TestDistributedTrainConfig,
    }
    if name in stage_name_to_config_map:
        config_dict = config_dict or {}
        return stage_name_to_config_map[name](**config_dict)
    else:
        raise ValueError(f"Unknown stage {name}.")


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        config_dict = json.load(f)
        config = Config(
            gpus=config_dict.get("gpus", 0),
            precision=config_dict.get("precision", 32),
            override_saved_config=config_dict.get("override_saved_config", None),
            deepspeed=config_dict.get("deepspeed", False),
            deepspeed_configs=config_dict.get("deepspeed_configs", None),
            early_stopping_patience=config_dict.get("early_stopping_patience", 100),
            working_directory=config_dict["working_directory"],
        )
        for s, c in zip(config_dict["stages"], config_dict["configs"]):
            config.stages.append(s)
            config.configs.append(stage_name_to_config(s, c))
        return config


def fix_missing(config):
    default = type(config)()
    for k, v in default.__dict__.items():
        if not hasattr(config, k):
            setattr(config, k, v)
    return config


def generate_config(stages: List[str], path: str, print_config: bool = True):
    config = Config()
    for stage in stages:
        config.stages.append(stage)
        config.configs.append(stage_name_to_config(stage))

    if print_config:
        pprint(config.dict())
    else:
        save_config(config, path)
        print(f"Config saved to {path}")


def save_config(config: Config, path: str):
    with open(path, "w") as f:
        json.dump(config.dict(), f, indent=4, sort_keys=True)
