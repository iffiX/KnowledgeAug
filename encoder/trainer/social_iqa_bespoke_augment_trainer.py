import os
from encoder.dataset.social_iqa import SocialIQABaseDataset
from encoder.utils.file import TorchCache
from encoder.utils.config import SocialIQABespokeAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .bespoke_base_trainer import BespokeBaseTrainer
from .social_iqa_augment_trainer import SocialIQAAugmentTrainer


class SocialIQABespokeAugmentTrainer(BespokeBaseTrainer, SocialIQAAugmentTrainer):
    def __init__(
        self,
        config: SocialIQABespokeAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        BespokeBaseTrainer.__init__(self, config=config)
        SocialIQAAugmentTrainer.__init__(
            self,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )

    def load_similarity_embedding(self):
        def generate():
            dataset = SocialIQABaseDataset(tokenizer=None)
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
            os.path.join(preprocess_cache_dir, f"social_iqa_similarity_embedding.pt"),
            generate_func=generate,
        ) as cache:
            return cache.data

    def load_model_state_dict(self):
        augment_trainer = SocialIQAAugmentTrainer.load_from_checkpoint(
            self.config.bespoke_base_checkpoint, map_location="cpu"
        )
        return augment_trainer.model.state_dict()
