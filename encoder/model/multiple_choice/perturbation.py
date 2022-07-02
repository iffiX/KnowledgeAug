# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Adapted from MT-DNN https://github.com/namisan/mt-dnn/blob/master/mt_dnn/perturbation.py
# Copyright (c) Microsoft. All rights reserved.

import torch
import logging
from .loss import stable_kl

logger = logging.getLogger(__name__)


def generate_noise(embed, mask, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise


class SmartPerturbation:
    """
    Implements "SMART: Robust and Efficient Fine-Tuning for Pre-trained
                Natural Language Models through Principled Regularized
                Optimization"
    https://arxiv.org/pdf/1911.03437.pdf
    """

    def __init__(
        self,
        epsilon=1e-6,
        step_size=1e-3,
        noise_var=1e-5,
        norm_p="inf",
        k=1,
        loss_map=[],
        norm_level=0,
    ):
        super(SmartPerturbation, self).__init__()
        self.epsilon = epsilon
        # eta
        self.step_size = step_size
        self.K = k
        # sigma
        self.noise_var = noise_var
        self.norm_p = norm_p
        self.loss_map = loss_map
        self.norm_level = norm_level > 0
        assert len(loss_map) > 0

    def _norm_grad(self, grad, eff_grad=None, sentence_level=False):
        if self.norm_p == "l2":
            if sentence_level:
                direction = grad / (
                    torch.norm(grad, dim=(-2, -1), keepdim=True) + self.epsilon
                )
            else:
                direction = grad / (
                    torch.norm(grad, dim=-1, keepdim=True) + self.epsilon
                )
        elif self.norm_p == "l1":
            direction = grad.sign()
        else:
            if sentence_level:
                direction = grad / (
                    grad.abs().max((-2, -1), keepdim=True)[0] + self.epsilon
                )
            else:
                direction = grad / (grad.abs().max(-1, keepdim=True)[0] + self.epsilon)
        return direction, None

    def forward(
        self, model, logits, input_ids, attention_mask, token_type_ids, *wargs, **kwargs
    ):
        # adv training
        embed = model.embed_encode(input_ids, token_type_ids, attention_mask)
        noise = generate_noise(embed, attention_mask, epsilon=self.noise_var)
        for step in range(0, self.K):
            adv_logits = model._forward(
                input_ids,
                attention_mask,
                token_type_ids,
                *wargs,
                inputs_embeds=embed + noise,
                **kwargs,
            )
            adv_loss = stable_kl(adv_logits, logits.detach(), reduce=False)
            (delta_grad,) = torch.autograd.grad(
                adv_loss, noise, only_inputs=True, retain_graph=False
            )
            norm = delta_grad.norm()
            if torch.isnan(norm) or torch.isinf(norm):
                return None, norm, None
            eff_delta_grad = delta_grad * self.step_size
            delta_grad = noise + delta_grad * self.step_size
            noise, _ = self._norm_grad(
                delta_grad, eff_grad=eff_delta_grad, sentence_level=self.norm_level
            )
            noise = noise.detach()
            noise.requires_grad_()
        adv_logits = model._forward(
            input_ids, attention_mask, token_type_ids, inputs_embeds=embed + noise,
        )
        adv_lc = self.loss_map[0]
        adv_loss = adv_lc(logits, adv_logits, ignore_index=-1, reduction="none")
        if len(adv_loss.size()) == 2:
            adv_loss = adv_loss.sum(dim=-1)
        return adv_loss, norm, adv_logits

    def loss(
        self, logits, _forward, *wargs, **kwargs,
    ):
        pass
