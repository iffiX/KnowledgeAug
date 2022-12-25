import os
from encoder.dataset.commonsense_qa2 import CommonsenseQA2BaseDataset
from encoder.utils.file import TorchCache
from encoder.utils.config import CommonsenseQA2BespokeAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .bespoke_base_trainer import BespokeBaseTrainer
from .commonsense_qa2_augment_trainer import CommonsenseQA2AugmentTrainer


class CommonsenseQA2BespokeAugmentTrainer(
    BespokeBaseTrainer, CommonsenseQA2AugmentTrainer
):
    def __init__(
        self,
        config: CommonsenseQA2BespokeAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        BespokeBaseTrainer.__init__(self, config=config)
        CommonsenseQA2AugmentTrainer.__init__(
            self,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )

    def load_similarity_embedding(self):
        def generate():
            dataset = CommonsenseQA2BaseDataset(tokenizer=None)
            texts = []
            ids = []
            for d in dataset.train_data + dataset.validate_data + dataset.test_data:
                # texts.append(d["text_question"] + " " + d["text_choices"])
                texts.append(d["text_question"])
                ids.append(d["id"])
            return self.compute_similarity_embedding(
                texts, ids, len(dataset.train_data)
            )

        with TorchCache(
            os.path.join(
                preprocess_cache_dir, f"commonsense_qa2_similarity_embedding.pt"
            ),
            generate_func=generate,
        ) as cache:
            return cache.data

    def load_model_state_dict(self):
        augment_trainer = CommonsenseQA2AugmentTrainer.load_from_checkpoint(
            self.config.bespoke_base_checkpoint, map_location="cpu"
        )
        return augment_trainer.model.state_dict()
