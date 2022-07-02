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


class CommonsenseQASearchTrainConfig(CommonsenseQATrainConfig):
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


class OpenBookQASearchTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False

    epochs_per_retriever_self_learn: int = 10
    epochs_per_reranker_self_learn: int = 10
    epochs: int = 80

    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None

    retriever_batch_size: int = 1
    retriever_accumulate_grad_batches: int = 32
    reranker_batch_size: int = 1
    reranker_accumulate_grad_batches: int = 32

    retriever_optimizer_class: str = "AdamW"
    retriever_learning_rate: float = 5e-5
    retriever_l2_regularization: float = 0
    retriever_scheduler_warmup_proportion: float = 0
    retriever_scheduler_cycles: int = 1

    reranker_optimizer_class: str = "AdamW"
    reranker_learning_rate: float = 5e-5
    reranker_l2_regularization: float = 0
    reranker_scheduler_warmup_proportion: float = 0
    reranker_scheduler_cycles: int = 1

    retriever_base_type: str = "sentence-transformers/all-mpnet-base-v2"
    retriever_model_configs: Optional[dict] = None
    retriever_max_seq_length: int = 128
    retriever_negative_samples: int = 4
    retriever_top_k: int = 50

    reranker_base_type: str = "microsoft/deberta-v3-large"
    reranker_model_configs: Optional[dict] = None
    reranker_max_seq_length: int = 128
    reranker_negative_samples: int = 4

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class OpenBookQAGailTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 20
    pretrain_epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    actor_pretrain_batch_size: int = 16
    rl_batch_size: int = 2

    optimizer_class: str = "AdamW"
    pretrain_learning_rate: float = 5e-5
    rl_actor_learning_rate: float = 5e-5
    rl_critic_learning_rate: float = 5e-5
    rl_discriminator_learning_rate: float = 5e-5

    base_type: str = "microsoft/deberta-v3-base"
    embedder_base_type: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 128
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    match_closest_when_no_equal: bool = True

    replay_size: int = 10000
    max_steps: int = 5
    state_delimeter: str = ", "


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


class ARCSearchTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False

    epochs_per_retriever_self_learn: int = 10
    epochs_per_reranker_self_learn: int = 10
    epochs: int = 80

    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None

    retriever_batch_size: int = 1
    retriever_accumulate_grad_batches: int = 32
    reranker_batch_size: int = 1
    reranker_accumulate_grad_batches: int = 32

    retriever_optimizer_class: str = "AdamW"
    retriever_learning_rate: float = 5e-5
    retriever_l2_regularization: float = 0
    retriever_scheduler_warmup_proportion: float = 0
    retriever_scheduler_cycles: int = 1

    reranker_optimizer_class: str = "AdamW"
    reranker_learning_rate: float = 5e-5
    reranker_l2_regularization: float = 0
    reranker_scheduler_warmup_proportion: float = 0
    reranker_scheduler_cycles: int = 1

    retriever_base_type: str = "sentence-transformers/all-mpnet-base-v2"
    retriever_model_configs: Optional[dict] = None
    retriever_max_seq_length: int = 128
    retriever_negative_samples: int = 4
    retriever_top_k: int = 50

    reranker_base_type: str = "microsoft/deberta-v3-large"
    reranker_model_configs: Optional[dict] = None
    reranker_max_seq_length: int = 128
    reranker_negative_samples: int = 4

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class EnsembleTrainConfig(BaseModel):
    task_trainer_stage: str
    checkpoints: List[str]
    matcher_modes_list: List[List[str]]
    matcher_seeds_list: List[List[int]]
    matcher_configs_list: List[List[dict]]


class Config(BaseModel):
    # Cuda ids of GPUs
    gpus: Optional[Union[int, List[int]]] = [0]
    precision: Optional[Union[int, str]] = 32
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
            CommonsenseQASearchTrainConfig,
            OpenBookQATrainConfig,
            OpenBookQASearchTrainConfig,
            ARCTrainConfig,
            ARCSearchTrainConfig,
            EnsembleTrainConfig,
        ]
    ] = []


def stage_name_to_config(name: str, config_dict: dict = None):
    stage_name_to_config_map = {
        "commonsense_qa": CommonsenseQATrainConfig,
        "commonsense_qa_search": CommonsenseQASearchTrainConfig,
        "openbook_qa": OpenBookQATrainConfig,
        "openbook_qa_search": OpenBookQASearchTrainConfig,
        "openbook_qa_gail": OpenBookQAGailTrainConfig,
        "arc": ARCTrainConfig,
        "arc_search": ARCSearchTrainConfig,
        "ensemble": EnsembleTrainConfig,
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
