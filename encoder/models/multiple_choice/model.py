# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Adapted from KEAR https://github.com/microsoft/KEAR/blob/main/model/model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import AutoModel, AutoModelForSequenceClassification
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from .layers import ChoicePredictor, ChoiceClassifier
from .perturbation import SmartPerturbation
from .sift import SiFTAdversarialLearner, hook_sift_layer
from .loss import LOSS_REGISTRY, LossCriterion


class ModelOutput:
    pass


class Model(nn.Module):
    """
    BERT style classifier, with VAT (Virtual adversarial training)

    Input format is [CLS] Question [SEP] Answer_A/B/C/D... [SEP]

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """

    def __init__(
        self,
        base_type,
        choice_num,
        batch_size: int = 32,
        adversarial_training: bool = False,
        adversarial_training_use_sift: bool = False,
        adversarial_training_config: dict = None,
        choice_predictor_type: str = "predictor",
        choice_predictor_dropout_prob: float = 0.0,
        choice_predictor_dataset_names: List[str] = None,
        regularize_with_bce_loss: bool = False,
        bce_loss_weight: float = 0.02,
    ):
        super(Model, self).__init__()

        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.batch_size = batch_size
        if choice_predictor_type not in ("classifier", "predictor", "mixed"):
            raise ValueError(f"Invalid choice_predictor_type {choice_predictor_type}")
        choice_predictor_cls = (
            ChoiceClassifier
            if choice_predictor_type == "classifier"
            else ChoicePredictor
        )
        if choice_predictor_dataset_names is None:
            choice_predictors = {
                "default": choice_predictor_cls(
                    self.base.config.hidden_size,
                    choice_num,
                    choice_predictor_dropout_prob,
                )
            }
        else:
            choice_predictors = {
                dataset: choice_predictor_cls(
                    self.base.config.hidden_size,
                    choice_num,
                    choice_predictor_dropout_prob,
                )
                for dataset in choice_predictor_dataset_names
            }
        if choice_predictor_type == "mixed":
            reranker = {}
            for name in choice_predictors:
                reranker[name + "_reranker"] = ChoiceClassifier(
                    self.base.config.hidden_size,
                    choice_num,
                    choice_predictor_dropout_prob,
                )
            choice_predictors.update(reranker)
        self.choice_predictor_type = choice_predictor_type
        self.choice_predictors = nn.ModuleDict(choice_predictors)
        self.adversarial_training = adversarial_training
        self.adversarial_training_use_sift = adversarial_training_use_sift
        self.adversarial_training_config = adversarial_training_config
        self.regularize_with_bce_loss = regularize_with_bce_loss
        self.bce_loss_weight = bce_loss_weight
        if adversarial_training:
            if adversarial_training_use_sift:
                adv_modules = hook_sift_layer(
                    self,
                    hidden_size=self.hidden_size,
                    learning_rate=adversarial_training_config["adv_step_size"],
                    init_perturbation=adversarial_training_config["adv_noise_var"],
                )
                self.adv_teacher = SiFTAdversarialLearner(self, adv_modules)
            else:
                # Use SMART instead
                cs = adversarial_training_config["adv_loss"]
                lc = LOSS_REGISTRY[LossCriterion[cs]](name=f"Adv Loss func: {cs}")
                self.adv_task_loss_criterion = [lc]
                self.adv_teacher = SmartPerturbation(
                    adversarial_training_config["adv_epsilon"],
                    adversarial_training_config["adv_step_size"],
                    adversarial_training_config["adv_noise_var"],
                    adversarial_training_config["adv_p_norm"],
                    adversarial_training_config["adv_k"],
                    loss_map=self.adv_task_loss_criterion,
                    norm_level=adversarial_training_config["adv_norm_level"],
                )

    def embed_encode(self, input_ids, token_type_ids=None, _attention_mask=None):
        """
        This method is used by Smart (SmartPerturbation)

        input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        embedding_output = self.base.embeddings(flat_input_ids, flat_token_type_ids)
        return embedding_output

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
        dataset_name=None,
        choice_mask=None,
    ):
        """

        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            labels: LongTensor of shape [batch_size,]
            dataset_name: String indicating which dataset it is from
            choice_mask: FloatTensor of shape [batch_size, choice_num]

        Returns:
            loss: Loss of training.
            right_num: Integer number of right predictions
            logits: Masked (If choice mask is present) logits, FloatTensor of shape [batch_size, choice_num]
            adv_norm: Norm of adversarial training
        """
        if self.choice_predictor_type == "mixed":
            logits, reranked_logits = self._forward(
                input_ids, attention_mask, token_type_ids, with_rerank=True
            )

            if choice_mask is not None:
                clf_logits = choice_mask * -1e7 + logits
                clf_reranked_logits = choice_mask * -1e7 + reranked_logits
            else:
                clf_logits = logits
                clf_reranked_logits = reranked_logits
            rank_loss = F.cross_entropy(
                clf_logits, labels.view(-1), reduction="none"
            ) + F.cross_entropy(clf_reranked_logits, labels.view(-1), reduction="none")
        else:
            logits = self._forward(input_ids, attention_mask, token_type_ids)

            # Masking for multi-task training where choice number is different for each dataset
            if choice_mask is not None:
                clf_logits = choice_mask * -1e7 + logits
            else:
                clf_logits = logits

            rank_loss = F.cross_entropy(clf_logits, labels.view(-1), reduction="none")

        if self.regularize_with_bce_loss:
            target = torch.zeros_like(clf_logits)
            target[range(clf_logits.shape[0]), labels.view(-1)] = 1
            label_loss = F.binary_cross_entropy_with_logits(
                clf_logits, target, reduction="none"
            ).mean(dim=1)
            loss = (
                rank_loss * (1 - self.bce_loss_weight)
                + label_loss * self.bce_loss_weight
            )
        else:
            loss = rank_loss

        if self.adversarial_training and self.training:
            if self.adversarial_training_use_sift:
                adv_loss, adv_norm = self.adv_teacher.loss(
                    logits,
                    self._forward,
                    self.adversarial_training_config["grad_adv_loss"],
                    self.adversarial_training_config["adv_loss"],
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    dataset_name,
                )
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
            else:
                adv_loss, adv_norm, adv_logits = self.adv_teacher.forward(
                    self,
                    logits,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    dataset_name,
                )
                if adv_loss is None:
                    adv_loss = torch.zeros_like(loss)
                    adv_norm = adv_loss
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
        else:
            adv_norm = None

        with torch.no_grad():
            predicts = torch.argmax(clf_logits, dim=1)
            right_num = int(torch.sum(predicts == labels))

        result = ModelOutput()
        result.loss = loss.mean()
        result.right_num = right_num
        result.logits = clf_logits
        result.adv_norm = adv_norm
        return result

    def predict(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        dataset_name=None,
        choice_mask=None,
    ):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            dataset_name: String indicating which dataset it is from
        """
        if self.choice_predictor_type == "mixed":
            logits, reranked_logits = self._forward(
                input_ids,
                attention_mask,
                token_type_ids,
                dataset_name=dataset_name,
                with_rerank=True,
            )
            if choice_mask is not None:
                logits = choice_mask * -1e7 + logits
            score = torch.sigmoid(logits)
            for i, row in enumerate(score):
                if torch.sum(row > 0.1) > 1:
                    logits[i] = reranked_logits[i]
        else:
            logits = self._forward(
                input_ids, attention_mask, token_type_ids, dataset_name=dataset_name
            )
            if choice_mask is not None:
                logits = choice_mask * -1e7 + logits

        return logits.detach()

    def _forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        dataset_name=None,
        inputs_embeds=None,
        with_rerank=False,
    ):
        if inputs_embeds is None:
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            flat_input_ids = None

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        if flat_attention_mask.shape[0] >= self.batch_size:
            hidden_state = torch.zeros(
                [flat_token_type_ids.shape[0], self.base.config.hidden_size],
                device=flat_attention_mask.device,
            )
            for b in range(0, flat_attention_mask.shape[0], self.batch_size):
                model_input = {
                    "input_ids": flat_input_ids[b : b + self.batch_size]
                    if flat_input_ids is not None
                    else None,
                    "attention_mask": flat_attention_mask[b : b + self.batch_size],
                    "inputs_embeds": inputs_embeds[b : b + self.batch_size]
                    if inputs_embeds is not None
                    else None,
                }
                if token_type_ids is not None:
                    model_input["token_type_ids"] = flat_token_type_ids[
                        b : b + self.batch_size
                    ]
                hidden_state[b : b + self.batch_size] = self.base(
                    **model_input
                ).last_hidden_state[:, 0, :]

        else:
            model_input = {
                "input_ids": flat_input_ids,
                "attention_mask": flat_attention_mask,
                "inputs_embeds": inputs_embeds,
            }
            if token_type_ids is not None:
                model_input["token_type_ids"] = flat_token_type_ids
            hidden_state = self.base(**model_input).last_hidden_state[:, 0, :]

        dataset_name = dataset_name or "default"

        if with_rerank:
            if self.choice_predictor_type != "mixed":
                raise ValueError("choice_predictor_type is not mixed")
            return (
                self.choice_predictors[dataset_name](hidden_state),
                self.choice_predictors[dataset_name + "_reranker"](
                    hidden_state.detach()
                ),
            )
        else:
            return self.choice_predictors[dataset_name](hidden_state)


class ModelForReranker(nn.Module):
    """
    BERT style classifier, with VAT (Virtual adversarial training)

    Input format is [CLS] Question [SEP] Answer_A/B/C/D... [SEP]

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """

    def __init__(
        self,
        base_type,
        batch_size: int = 32,
        adversarial_training: bool = False,
        adversarial_training_use_sift: bool = False,
        adversarial_training_config: dict = None,
    ):
        super(ModelForReranker, self).__init__()

        self.base = AutoModelForSequenceClassification.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
            num_labels=1,
        )
        self.batch_size = batch_size
        self.adversarial_training = adversarial_training
        self.adversarial_training_use_sift = adversarial_training_use_sift
        self.adversarial_training_config = adversarial_training_config
        if adversarial_training:
            if adversarial_training_use_sift:
                adv_modules = hook_sift_layer(
                    self,
                    hidden_size=self.hidden_size,
                    learning_rate=adversarial_training_config["adv_step_size"],
                    init_perturbation=adversarial_training_config["adv_noise_var"],
                )
                self.adv_teacher = SiFTAdversarialLearner(self, adv_modules)
            else:
                # Use SMART instead
                cs = adversarial_training_config["adv_loss"]
                lc = LOSS_REGISTRY[LossCriterion[cs]](name=f"Adv Loss func: {cs}")
                self.adv_task_loss_criterion = [lc]
                self.adv_teacher = SmartPerturbation(
                    adversarial_training_config["adv_epsilon"],
                    adversarial_training_config["adv_step_size"],
                    adversarial_training_config["adv_noise_var"],
                    adversarial_training_config["adv_p_norm"],
                    adversarial_training_config["adv_k"],
                    loss_map=self.adv_task_loss_criterion,
                    norm_level=adversarial_training_config["adv_norm_level"],
                )

    def embed_encode(self, input_ids, token_type_ids=None, _attention_mask=None):
        """
        This method is used by Smart (SmartPerturbation)

        input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        embedding_output = self.base.embeddings(flat_input_ids, flat_token_type_ids)
        return embedding_output

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels, choice_mask=None,
    ):
        """

        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            labels: LongTensor of shape [batch_size,]
            choice_mask: FloatTensor of shape [batch_size, choice_num]

        Returns:
            loss: Loss of training.
            right_num: Integer number of right predictions
            logits: Masked (If choice mask is present) logits, FloatTensor of shape [batch_size, choice_num]
            adv_norm: Norm of adversarial training
        """
        logits = self._forward(input_ids, attention_mask, token_type_ids)

        # Masking for multi-task training where choice number is different for each dataset
        if choice_mask is not None:
            clf_logits = choice_mask * -1e7 + logits
        else:
            clf_logits = logits

        loss = F.cross_entropy(clf_logits, labels.view(-1), reduction="none")

        if self.adversarial_training and self.training:
            if self.adversarial_training_use_sift:
                adv_loss, adv_norm = self.adv_teacher.loss(
                    logits,
                    self._forward,
                    self.adversarial_training_config["grad_adv_loss"],
                    self.adversarial_training_config["adv_loss"],
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
            else:
                adv_loss, adv_norm, adv_logits = self.adv_teacher.forward(
                    self, logits, input_ids, attention_mask, token_type_ids,
                )
                if adv_loss is None:
                    adv_loss = torch.zeros_like(loss)
                    adv_norm = adv_loss
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
        else:
            adv_norm = None

        with torch.no_grad():
            predicts = torch.argmax(clf_logits, dim=1)
            right_num = int(torch.sum(predicts == labels))

        result = ModelOutput()
        result.loss = loss.mean()
        result.right_num = right_num
        result.logits = clf_logits
        result.adv_norm = adv_norm
        return result

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        """
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        return logits.detach()

    def _forward(
        self, input_ids, attention_mask, token_type_ids, inputs_embeds=None,
    ):
        if inputs_embeds is None:
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            flat_input_ids = None

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        if flat_attention_mask.shape[0] >= self.batch_size:
            logits = torch.zeros(
                [flat_token_type_ids.shape[0], 1], device=flat_attention_mask.device,
            )
            for b in range(0, flat_attention_mask.shape[0], self.batch_size):
                model_input = {
                    "input_ids": flat_input_ids[b : b + self.batch_size]
                    if input_ids is not None
                    else None,
                    "attention_mask": flat_attention_mask[b : b + self.batch_size],
                    "inputs_embeds": inputs_embeds[b : b + self.batch_size]
                    if inputs_embeds is not None
                    else None,
                }
                if token_type_ids is not None:
                    model_input["token_type_ids"] = flat_token_type_ids[
                        b : b + self.batch_size
                    ]

                logits[b : b + self.batch_size] = self.base(**model_input).logits

        else:
            model_input = {
                "input_ids": flat_input_ids,
                "attention_mask": flat_attention_mask,
                "inputs_embeds": inputs_embeds,
            }
            if token_type_ids is not None:
                model_input["token_type_ids"] = flat_token_type_ids
            logits = self.base(**model_input).logits
        return logits.view(token_type_ids.shape[0], token_type_ids.shape[1])


class ModelForRetriever(nn.Module):
    def __init__(
        self,
        base_type,
        batch_size: int = 32,
        adversarial_training: bool = False,
        adversarial_training_use_sift: bool = False,
        adversarial_training_config: dict = None,
    ):
        super(ModelForRetriever, self).__init__()
        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.batch_size = batch_size
        self.adversarial_training = adversarial_training
        self.adversarial_training_use_sift = adversarial_training_use_sift
        self.adversarial_training_config = adversarial_training_config
        if adversarial_training:
            if adversarial_training_use_sift:
                adv_modules = hook_sift_layer(
                    self,
                    hidden_size=self.hidden_size,
                    learning_rate=adversarial_training_config["adv_step_size"],
                    init_perturbation=adversarial_training_config["adv_noise_var"],
                )
                self.adv_teacher = SiFTAdversarialLearner(self, adv_modules)
            else:
                # Use SMART instead
                cs = adversarial_training_config["adv_loss"]
                lc = LOSS_REGISTRY[LossCriterion[cs]](name=f"Adv Loss func: {cs}")
                self.adv_task_loss_criterion = [lc]
                self.adv_teacher = SmartPerturbation(
                    adversarial_training_config["adv_epsilon"],
                    adversarial_training_config["adv_step_size"],
                    adversarial_training_config["adv_noise_var"],
                    adversarial_training_config["adv_p_norm"],
                    adversarial_training_config["adv_k"],
                    loss_map=self.adv_task_loss_criterion,
                    norm_level=adversarial_training_config["adv_norm_level"],
                )

    def embed_encode(self, input_ids, token_type_ids=None, _attention_mask=None):
        """
        This method is used by Smart (SmartPerturbation)

        input_ids: LongTensor of shape [batch_size, 1 + compare_num, seq_length]
        token_type_ids: LongTensor of shape [batch_size, 1 + compare_num, seq_length]
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        embedding_output = self.base.embeddings(flat_input_ids, flat_token_type_ids)
        return embedding_output

    def forward(
        self, input_ids, attention_mask, token_type_ids, labels,
    ):
        """

        Args:
            input_ids: LongTensor of shape [batch_size, 1 + compare_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, 1 + compare_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, 1 + compare_num, seq_length] or None
            labels: LongTensor of shape [batch_size,]

        Returns:
            loss: Loss of training.
            right_num: Integer number of right predictions
            logits: Logits before softmax, FloatTensor of shape [batch_size, 1 + compare_num]
            adv_norm: Norm of adversarial training
        """
        logits = self._forward(input_ids, attention_mask, token_type_ids)
        loss = F.cross_entropy(logits * 20, labels.view(-1), reduction="none")

        if self.adversarial_training and self.training:
            if self.adversarial_training_use_sift:
                adv_loss, adv_norm = self.adv_teacher.loss(
                    logits,
                    self._forward,
                    self.adversarial_training_config["grad_adv_loss"],
                    self.adversarial_training_config["adv_loss"],
                    input_ids,
                    attention_mask,
                    token_type_ids,
                )
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
            else:
                adv_loss, adv_norm, adv_logits = self.adv_teacher.forward(
                    self, logits, input_ids, attention_mask, token_type_ids,
                )
                if adv_loss is None:
                    adv_loss = torch.zeros_like(loss)
                    adv_norm = adv_loss
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
        else:
            adv_norm = None

        with torch.no_grad():
            predicts = torch.argmax(logits, dim=1)
            right_num = int(torch.sum(predicts == labels))

        result = ModelOutput()
        result.loss = loss.mean()
        result.right_num = right_num
        result.logits = logits
        result.adv_norm = adv_norm
        return result

    def predict(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, 1 + compare_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, 1 + compare_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, 1 + compare_num, seq_length]
        """
        logits = self._forward(input_ids, attention_mask, token_type_ids,)
        return logits.detach()

    def predict_embedding(self, input_ids, attention_mask, token_type_ids=None):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, predict_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, predict_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, predict_num, seq_length]
        """
        embeds = self._forward(
            input_ids, attention_mask, token_type_ids, return_embeds=True
        )
        return embeds.detach()

    def _forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        inputs_embeds=None,
        return_embeds=False,
    ):
        batch_size = attention_mask.shape[0]
        if inputs_embeds is None:
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            flat_input_ids = None

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = None
        if token_type_ids is not None:
            flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        if flat_attention_mask.shape[0] >= self.batch_size:
            sub_hidden_state_list = []
            for b in range(0, flat_attention_mask.shape[0], self.batch_size):
                model_input = {
                    "input_ids": flat_input_ids[b : b + self.batch_size]
                    if input_ids is not None
                    else None,
                    "attention_mask": flat_attention_mask[b : b + self.batch_size],
                    "inputs_embeds": inputs_embeds[b : b + self.batch_size]
                    if inputs_embeds is not None
                    else None,
                }
                if token_type_ids is not None:
                    model_input["token_type_ids"] = flat_token_type_ids[
                        b : b + self.batch_size
                    ]

                sub_hidden_state_list.append(self.base(**model_input).last_hidden_state)
            hidden_state = torch.cat(sub_hidden_state_list)
        else:
            model_input = {
                "input_ids": flat_input_ids,
                "attention_mask": flat_attention_mask,
                "inputs_embeds": inputs_embeds,
            }
            if token_type_ids is not None:
                model_input["token_type_ids"] = flat_token_type_ids
            hidden_state = self.base(**model_input).last_hidden_state
        # h = hidden_state[:, 0, :]
        h = self.mean_pooling(hidden_state, flat_attention_mask)
        h = F.normalize(h.view(batch_size, -1, h.shape[1]), p=2, dim=2)
        if return_embeds:
            return h
        else:
            return (h[:, :1, :] * h[:, 1:, :]).sum(-1)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
