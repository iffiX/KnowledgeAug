import json
from pydantic import BaseModel
from pprint import pprint
from typing import *


class AugmentBaseConfig(BaseModel):
    skip: bool = False
    load: bool = False
    seed: int = 42
    save: Union[bool, int] = True
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
    device_map: Optional[Dict[int, List[int]]] = None


class BespokeBaseConfig(BaseModel):
    bespoke_optimizer_class: str = "AdamW"
    bespoke_learning_rate: float = 5e-6
    bespoke_l2_regularization: float = 0
    bespoke_top_k: int = 12
    bespoke_min_similarity: float = 0.58
    bespoke_iterations: int = 2
    bespoke_batch_size: int = 4
    bespoke_accumulate_grad_batches: int = 1
    bespoke_base_checkpoint: str = ""


class SampleBaseConfig(BaseModel):
    skip: bool = False
    load: bool = False
    seed: int = 0
    save: Union[bool, int] = True
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

    base_type: str = "microsoft/deberta-v3-large"
    pad_by_longest: bool = True
    max_seq_length: Union[int, None] = None


class OpenBookQASingleChoiceSampleTrainConfig(SampleBaseConfig):
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_steps: int = 2
    max_depth: int = 2
    beam_size: int = 1
    return_beam_num: int = 1
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    expand_choice_num: int = 3
    inference_batch_size: int = 128
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    wrong_choice: str = "WRONG_CHOICE"
    negative_samples: int = 31
    negative_shuffle_seed: int = 42


class OpenBookQAMultipleChoiceSampleTrainConfig(SampleBaseConfig):
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_steps: int = 2
    max_depth: int = 2
    beam_size: int = 5
    return_beam_num: int = 5
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    inference_batch_size: int = 128
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    negative_samples: int = 31
    negative_shuffle_seed: int = 42


class OpenBookQAAugmentTrainConfig(AugmentBaseConfig):
    max_seq_length: int = 256
    generate_length: int = 20
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_depth: int = 2
    use_augment: bool = True
    augment_method: str = "standard"
    augment_use_parts: str = "all"
    sample_type: str = "mc"


class OpenBookQABespokeAugmentTrainConfig(
    BespokeBaseConfig, OpenBookQAAugmentTrainConfig
):
    pass


class QASCSingleChoiceSampleTrainConfig(SampleBaseConfig):
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_steps: int = 3
    max_depth: int = 2
    beam_size: int = 1
    return_beam_num: int = 1
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    expand_choice_num: int = 4
    inference_batch_size: int = 128
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    wrong_choice: str = "WRONG_CHOICE"
    negative_samples: int = 3
    negative_shuffle_seed: int = 42


class QASCMultipleChoiceSampleTrainConfig(SampleBaseConfig):
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_steps: int = 3
    max_depth: int = 2
    beam_size: int = 4
    return_beam_num: int = 4
    min_logits: Union[float, None] = None
    max_inference_num: int = 50000
    inference_batch_size: int = 128
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    negative_samples: int = 3
    negative_shuffle_seed: int = 42


class QASCAugmentTrainConfig(AugmentBaseConfig):
    max_seq_length: int = 256
    generate_length: int = 20
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_depth: int = 2
    use_augment: bool = True
    augment_method: str = "raw_decode"
    augment_use_parts: str = "all"
    sample_type: str = "sc"


class QASCBespokeAugmentTrainConfig(BespokeBaseConfig, QASCAugmentTrainConfig):
    pass


class CommonsenseQA2AugmentTrainConfig(AugmentBaseConfig):
    max_seq_length: int = 256
    generate_length: int = 20
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_depth: int = 2
    use_augment: bool = True
    augment_method: str = "raw_decode"
    augment_use_parts: str = "all"


class CommonsenseQA2BespokeAugmentTrainConfig(
    BespokeBaseConfig, CommonsenseQA2AugmentTrainConfig
):
    pass


class SocialIQASingleChoiceSampleTrainConfig(SampleBaseConfig):
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_steps: int = 2
    max_depth: int = 2
    beam_size: int = 3
    return_beam_num: int = 3
    min_logits: Union[float, None] = None
    max_inference_num: int = 20000
    expand_choice_num: int = 3
    inference_batch_size: int = 128
    state_delimeter: str = ", "
    end_of_reasoning: str = "END_OF_REASONING"
    wrong_choice: str = "WRONG_CHOICE"
    negative_samples: int = 3
    negative_shuffle_seed: int = 42


class SocialIQAAugmentTrainConfig(AugmentBaseConfig):
    max_seq_length: int = 256
    generate_length: int = 20
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    max_depth: int = 2
    use_augment: bool = True
    augment_method: str = "raw_decode"
    augment_use_parts: str = "all"


class SocialIQABespokeAugmentTrainConfig(
    BespokeBaseConfig, SocialIQAAugmentTrainConfig
):
    pass


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
            OpenBookQASingleChoiceSampleTrainConfig,
            OpenBookQAMultipleChoiceSampleTrainConfig,
            OpenBookQAAugmentTrainConfig,
            QASCSingleChoiceSampleTrainConfig,
            QASCMultipleChoiceSampleTrainConfig,
            QASCAugmentTrainConfig,
            CommonsenseQA2AugmentTrainConfig,
            SocialIQASingleChoiceSampleTrainConfig,
            SocialIQAAugmentTrainConfig,
        ]
    ] = []


def stage_name_to_config(name: str, config_dict: dict = None):
    stage_name_to_config_map = {
        "openbook_qa_sc_sample": OpenBookQASingleChoiceSampleTrainConfig,
        "openbook_qa_mc_sample": OpenBookQAMultipleChoiceSampleTrainConfig,
        "openbook_qa_augment": OpenBookQAAugmentTrainConfig,
        "openbook_qa_bespoke_augment": OpenBookQABespokeAugmentTrainConfig,
        "qasc_sc_sample": QASCSingleChoiceSampleTrainConfig,
        "qasc_mc_sample": QASCMultipleChoiceSampleTrainConfig,
        "qasc_augment": QASCAugmentTrainConfig,
        "qasc_bespoke_augment": QASCBespokeAugmentTrainConfig,
        "commonsense_qa2_augment": CommonsenseQA2AugmentTrainConfig,
        "commonsense_qa2_bespoke_augment": CommonsenseQA2BespokeAugmentTrainConfig,
        "social_iqa_sc_sample": SocialIQASingleChoiceSampleTrainConfig,
        "social_iqa_augment": SocialIQAAugmentTrainConfig,
        "social_iqa_bespoke_augment": SocialIQABespokeAugmentTrainConfig,
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
        assert len(config_dict["stages"]) == len(
            config_dict["configs"]
        ), "Invalid config, pipeline stage number must be equal to the number of stage configs."
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
