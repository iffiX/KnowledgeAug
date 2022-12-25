import copy
import torch as t
from typing import Any
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.models.embedder import Embedder
from transformers import BatchEncoding


class BespokeBaseTrainer:
    def __init__(
        self, config: Any, **__,
    ):
        self.config = config
        self.similarity_embedding = self.load_similarity_embedding()
        self.model_state_dict = self.load_model_state_dict()
        self.model_parameters_loaded = False

    def on_train_start(self):
        raise RuntimeError(
            "Bespoke trainer cannot be trained, run validate or test instead."
        )

    def validation_or_test_step(self, batch: BatchEncoding, *_, **__):
        if "t5-" in self.config.base_type:
            raise RuntimeError("bespoke is not implemented for t5 models")

        else:
            if not self.model_parameters_loaded:
                self.model.load_state_dict(self.model_state_dict)
                self.model_parameters_loaded = True
            model = copy.deepcopy(self.model)
            model.train()

            optim_cls = getattr(t.optim, self.config.bespoke_optimizer_class)
            optim = optim_cls(
                model.parameters(),
                lr=self.config.bespoke_learning_rate,
                weight_decay=self.config.bespoke_l2_regularization,
            )

            # pytorch lightning disables grad during validation by default
            with t.enable_grad():
                similarity = t.sum(
                    self.similarity_embedding[2][batch["id"][0]]
                    * self.similarity_embedding[1],
                    dim=1,
                )
                most_similar_train_indices = t.topk(
                    similarity, k=self.config.bespoke_top_k
                ).indices
                for i in range(len(most_similar_train_indices)):
                    if (
                        similarity[most_similar_train_indices[i]]
                        < self.config.bespoke_min_similarity
                    ):
                        break

                batch_num = 0
                optim.zero_grad()
                for _ in range(self.config.bespoke_iterations):
                    if i > 0:
                        most_similar_train_indices = most_similar_train_indices[:i]

                        for b in range(
                            0,
                            len(most_similar_train_indices),
                            self.config.bespoke_batch_size,
                        ):
                            train_batch = collate_function_dict_to_batch_encoding(
                                [
                                    self.dataset.train_dataset[x]
                                    for x in most_similar_train_indices[
                                        b : b + self.config.bespoke_batch_size
                                    ]
                                ],
                            ).to(self.real_device)
                            batch_num += 1

                            loss = (
                                model(
                                    input_ids=train_batch["sentence"],
                                    attention_mask=train_batch["mask"],
                                    token_type_ids=train_batch["type_ids"],
                                    labels=train_batch["label"],
                                ).loss
                                # / self.config.bespoke_batch_size
                            ) / self.config.bespoke_accumulate_grad_batches
                            loss.backward()

                            if (
                                batch_num % self.config.bespoke_accumulate_grad_batches
                                == 0
                            ):
                                optim.step()
                                optim.zero_grad()

            if batch_num % self.config.bespoke_accumulate_grad_batches != 0:
                optim.step()
                optim.zero_grad()

            model.eval()
            return {
                "batch": batch.to("cpu"),
                "result": model.predict(
                    input_ids=batch["sentence"].to(self.real_device),
                    attention_mask=batch["mask"].to(self.real_device),
                    token_type_ids=batch["type_ids"].to(self.real_device),
                ).to(device="cpu", dtype=t.float32),
            }

    def load_similarity_embedding(self):
        raise NotImplementedError()

    def load_model_state_dict(self):
        raise NotImplementedError()

    def compute_similarity_embedding(self, texts, ids, train_size):
        embedder = Embedder()
        embeddings = embedder.embed(texts)
        return (
            ids[:train_size],
            embeddings[:train_size],
            {
                data_id: data_h.unsqueeze(0)
                for data_id, data_h in zip(ids[train_size:], embeddings[train_size:])
            },
        )
