import os
from encoder.dataset.openbook_qa import OpenBookQABaseDataset
from encoder.utils.file import TorchCache
from encoder.utils.config import OpenBookQABespokeAugmentTrainConfig
from encoder.utils.settings import preprocess_cache_dir
from .bespoke_base_trainer import BespokeBaseTrainer
from .openbook_qa_augment_trainer import OpenBookQAAugmentTrainer


class OpenBookQABespokeAugmentTrainer(BespokeBaseTrainer, OpenBookQAAugmentTrainer):
    def __init__(
        self,
        config: OpenBookQABespokeAugmentTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        BespokeBaseTrainer.__init__(self, config=config)
        OpenBookQAAugmentTrainer.__init__(
            self,
            config=config,
            stage_result_path=stage_result_path,
            is_distributed=is_distributed,
        )
        # Must use original train data since similarity embeddings are computed using those.
        self.dataset.train_data = self.dataset.original_train_data

    def load_similarity_embedding(self):
        def generate():
            dataset = OpenBookQABaseDataset(tokenizer=None)
            texts = []
            ids = []
            for d in (
                dataset.original_train_data + dataset.validate_data + dataset.test_data
            ):
                # texts.append(d["text_question"] + " " + d["text_choices"])
                texts.append(d["text_question"])
                ids.append(d["id"])
            return self.compute_similarity_embedding(
                texts, ids, len(dataset.original_train_data)
            )

        with TorchCache(
            os.path.join(preprocess_cache_dir, f"openbook_qa_similarity_embedding.pt"),
            generate_func=generate,
        ) as cache:
            return cache.data

    def load_model_state_dict(self):
        augment_trainer = OpenBookQAAugmentTrainer.load_from_checkpoint(
            self.config.bespoke_base_checkpoint, map_location="cpu"
        )
        return augment_trainer.model.state_dict()
