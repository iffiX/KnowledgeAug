import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from torch.distributions import Normal
from transformers import AutoModel, AutoTokenizer
from machin.model.nets import NeuralNetworkModule
from machin.frame.algorithms import PPO as _PPO, TRPO as _TRPO
from machin.frame.algorithms.trpo import default_logger, safe_call, safe_return
from machin.model.algorithms.trpo import ActorContinuous
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)


class MovableString(str):
    def to(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    @property
    def shape(self):
        return 0

    @property
    def dtype(self):
        return "str"

    def equal(self, str):
        return self == str


MStr = MovableString


class Actor(ActorContinuous):
    def __init__(self, base_type, action_dim, max_length: Optional[int] = None):
        base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            truncation_side="left",
        )
        super(Actor, self).__init__(action_dim, log_std=-1)
        self.base = base
        self.fc = nn.Linear(self.base.config.hidden_size, action_dim)
        self.max_length = max_length
        self.is_cache_enabled = False
        self.cache = {}
        self.set_input_module(self.base)
        self.set_output_module(self.fc)

    def clear_cache(self):
        self.cache = {}

    def forward(
        self,
        state: List[MovableString],
        action_embedding: List[t.Tensor] = None,
        sample: bool = True,
    ):
        key = tuple(state)
        if not self.is_cache_enabled or key not in self.cache:
            batch_encoding = self.tokenizer(
                state,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.input_device)
            # out = self.base(**batch_encoding).last_hidden_state[:, 0, :]

            token_embeddings = self.base(**batch_encoding).last_hidden_state
            input_mask_expanded = (
                batch_encoding["attention_mask"]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            out = t.sum(token_embeddings * input_mask_expanded, 1) / t.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
            if self.is_cache_enabled:
                self.cache[key] = out.detach()
        else:
            out = self.cache[key]

        if sample:
            return self.sample(
                F.normalize(self.fc(out), p=2, dim=1),
                t.cat(action_embedding, dim=0)
                if action_embedding is not None
                else None,
            )
        else:
            return F.normalize(self.fc(out), p=2, dim=1)

    def parameters(self, recurse: bool = True):
        return self.fc.parameters(recurse=recurse)

    def all_parameters(self, recurse: bool = True):
        return super(Actor, self).parameters(recurse=recurse)

    def sample(self, mean: t.tensor, action=None):
        """
        You must call this function to sample an action and its log probability
        during forward().
        Args:
            mean: Probability tensor of shape ``[batch, action_num]``,
                usually produced by a softmax layer.
            action: The action to be evaluated. set to ``None`` if you are sampling
                a new batch of actions.
        Returns:
            Action tensor of shape ``[batch, action_dim]``,
            Action log probability tensor of shape ``[batch, 1]``.
        """
        self.action_param = mean
        dist = Normal(loc=mean, scale=t.exp(self.action_log_std))
        if action is None:
            action = dist.sample()
        return (
            action.to(self.input_device),
            dist.log_prob(action.to(self.input_device)).sum(dim=1, keepdims=True),
            dist.entropy(),
        )

    def compare_kl(self, params: t.tensor, *args, **kwargs):
        with t.no_grad():
            new_params = TRPO.get_flat_params(self)

            TRPO.set_flat_params(self, params)
            self.forward(*args, **kwargs)
            mean1 = self.action_param
            log_std1 = self.action_log_std.expand_as(mean1)
            std1 = t.exp(log_std1)

            TRPO.set_flat_params(self, new_params)
            self.forward(*args, **kwargs)
            mean0 = self.action_param
            log_std0 = self.action_log_std.expand_as(mean1)
            std0 = t.exp(log_std0)

            kl = (
                log_std1
                - log_std0
                + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2))
                - 0.5
            )
            return kl.sum(1).mean().item()


class Critic(NeuralNetworkModule):
    def __init__(self, base_type, max_length: Optional[int] = None):
        super(Critic, self).__init__()
        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            truncation_side="left",
        )
        self.fc = nn.Linear(self.base.config.hidden_size, 1)
        self.max_length = max_length
        self.set_input_module(self.base)
        self.set_output_module(self.fc)

    def forward(self, state: List[MovableString]):
        batch_encoding = self.tokenizer(
            state,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.input_device)
        return self.fc(self.base(**batch_encoding).last_hidden_state[:, 0, :])


class Discriminator(NeuralNetworkModule):
    def __init__(
        self, base_type, state_delimeter: str = ", ", max_length: Optional[int] = None
    ):
        super(Discriminator, self).__init__()
        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            truncation_side="left",
        )
        self.fc = nn.Linear(self.base.config.hidden_size, 1)
        self.max_length = max_length
        self.state_delimeter = state_delimeter
        self.set_input_module(self.base)
        self.set_output_module(self.fc)

    def forward(
        self,
        state: Union[MovableString, List[MovableString]],
        action_string: Union[MovableString, List[MovableString]] = None,
    ):
        if isinstance(state, MovableString):
            state = [state]
        if isinstance(action_string, MovableString):
            action_string = [action_string]
        batch_encoding = self.tokenizer(
            [st + self.state_delimeter + act for st, act in zip(state, action_string)],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.input_device)
        return t.sigmoid(
            self.fc(self.base(**batch_encoding).last_hidden_state[:, 0, :])
        )


class Embedder(NeuralNetworkModule):
    def __init__(
        self,
        base_type,
        max_length: int = None,
        pad_by_longest: bool = False,
        infer_batch_size: int = 256,
    ):
        super(Embedder, self).__init__()
        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            truncation_side="left",
        )
        self.max_length = max_length
        self.pad_by_longest = pad_by_longest
        self.infer_batch_size = infer_batch_size
        self.set_input_module(self.base)
        self.set_output_module(self.base)

    @property
    def output_dim(self) -> int:
        return self.base.config.hidden_size

    def forward(self, choice_string: List[str]):
        result_embedding = []
        for offset in range(0, len(choice_string), self.infer_batch_size):
            batch_encoding = self.tokenizer(
                choice_string[offset : offset + self.infer_batch_size],
                padding="max_length" if not self.pad_by_longest else "longest",
                max_length=self.max_length if not self.pad_by_longest else None,
                truncation=True,
                return_tensors="pt",
            ).to(self.input_device)
            token_embeddings = self.base(**batch_encoding)[0]
            input_mask_expanded = (
                batch_encoding["attention_mask"]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            pooled_embedding = t.sum(
                token_embeddings * input_mask_expanded, 1
            ) / t.clamp(input_mask_expanded.sum(1), min=1e-9)
            result_embedding.append(pooled_embedding)
        result_embedding = t.cat(result_embedding, dim=0)
        return F.normalize(result_embedding, p=2, dim=1)


class PPO(_PPO):
    def update(self, concatenate_samples=True, **__):
        # DOC INHERITED

        self.actor.train()
        self.critic.train()

        # sample a batch
        (
            batch_size,
            (state, action, target_value, advantage),
        ) = self.replay_buffer.sample_batch(
            self.batch_size,
            sample_method="random_unique",
            concatenate=concatenate_samples,
            sample_attrs=["state", "action", "value", "gae"],
            additional_concat_attrs=["value", "gae"],
        )
        if not concatenate_samples:
            target_value = t.Tensor(target_value).view(batch_size, 1)
            advantage = t.Tensor(advantage).view(batch_size, 1)

        # normalize advantage
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        # Infer original action log probability
        # TODO:
        #  This temporary fix is not efficient, maybe requires
        #  PPO store API alternation.
        with t.no_grad():
            __, action_log_prob, *_ = self._eval_act(state, action)
            action_log_prob = action_log_prob.view(batch_size, 1)

        if self.entropy_weight is not None:
            __, new_action_log_prob, new_action_entropy, *_ = self._eval_act(
                state, action
            )
        else:
            __, new_action_log_prob, *_ = self._eval_act(state, action)
            new_action_entropy = None

        new_action_log_prob = new_action_log_prob.view(batch_size, 1)

        # calculate surrogate loss
        # The function of this process is ignoring actions that are not
        # likely to be produced in current actor policy distribution,
        # Because in each update, the old policy distribution diverges
        # from the current distribution more and more.
        sim_ratio = t.exp(new_action_log_prob - action_log_prob)
        advantage = advantage.type_as(sim_ratio).to(sim_ratio.device)
        surr_loss_1 = sim_ratio * advantage
        surr_loss_2 = (
            t.clamp(sim_ratio, 1 - self.surr_clip, 1 + self.surr_clip) * advantage
        )

        # calculate policy loss using surrogate loss
        act_policy_loss = -t.min(surr_loss_1, surr_loss_2)

        if new_action_entropy is not None:
            act_policy_loss += self.entropy_weight * new_action_entropy.mean()

        act_policy_loss = act_policy_loss.mean()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        # calculate value loss
        value = self._criticize(state)
        value_loss = (
            self.criterion(target_value.type_as(value).to(value.device), value)
            * self.value_weight
        )

        if self.visualize:
            self.visualize_model(value_loss, "critic", self.visualize_dir)

        # Update actor and critic networks
        self.actor.zero_grad()
        self.critic.zero_grad()
        self._backward(act_policy_loss + value_loss)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_max)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
        self.actor_optim.step()
        self.critic_optim.step()

        self.replay_buffer.clear()
        self.actor.eval()
        self.critic.eval()
        return (
            -act_policy_loss.item(),
            value_loss.item(),
        )


class TRPO(_TRPO):
    def update(
        self, update_value=True, update_policy=True, concatenate_samples=True, **__
    ):
        # DOC INHERITED
        sum_value_loss = 0

        self.actor.train()
        self.critic.train()

        # sample a batch for actor training
        batch_size, (state, action, advantage) = self.replay_buffer.sample_batch(
            -1,
            sample_method="all",
            concatenate=concatenate_samples,
            sample_attrs=["state", "action", "gae"],
            additional_concat_attrs=["gae"],
        )

        if not concatenate_samples:
            advantage = t.Tensor(advantage).view(batch_size, 1)

        # normalize advantage
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

        # Train actor
        # define two closures needed by fvp functions
        with t.no_grad():
            ___, fixed_action_log_prob, *_ = self._eval_act(state, action)
            fixed_action_log_prob = fixed_action_log_prob.view(batch_size, 1)
            fixed_params = self.get_flat_params(self.actor)

        def actor_loss_func():
            ____, action_log_prob, *_ = self._eval_act(state, action)
            action_log_prob = action_log_prob.view(batch_size, 1)
            action_loss = -advantage.to(action_log_prob.device) * t.exp(
                action_log_prob - fixed_action_log_prob
            )
            return action_loss.mean()

        def actor_kl_func():
            state["params"] = fixed_params
            return safe_return(safe_call(self.actor, state, method="compare_kl"))

        act_policy_loss = actor_loss_func()

        if self.visualize:
            self.visualize_model(act_policy_loss, "actor", self.visualize_dir)

        # Update actor network
        if update_policy:

            def fvp(v):
                if self.hv_mode == "fim":
                    return self._fvp_fim(state, v, self.damping)
                else:
                    return self._fvp_direct(state, v, self.damping)

            loss_grad = self.get_flat_grad(
                act_policy_loss, list(self.actor.parameters())
            ).detach()

            # usually 1e-15 is low enough
            if t.allclose(loss_grad, t.zeros_like(loss_grad), atol=1e-15):
                default_logger.warning(
                    "TRPO detects zero gradient, update step skipped."
                )
                return 0, 0

            step_dir = self._conjugate_gradients(
                fvp,
                -loss_grad,
                eps=self.conjugate_eps,
                iterations=self.conjugate_iterations,
                res_threshold=self.conjugate_res_threshold,
            )

            # Maximum step size mentioned in appendix C of the paper.
            beta = np.sqrt(2 * self.kl_max_delta / step_dir.dot(fvp(step_dir)).item())

            full_step = step_dir * beta
            if not self._line_search(
                self.actor, actor_loss_func, actor_kl_func, full_step, self.kl_max_delta
            ):
                default_logger.warning(
                    "Cannot find an update step to satisfy kl_max_delta, "
                    "consider increase line_search_backtracks"
                )

        for _ in range(self.critic_update_times):
            # sample a batch
            batch_size, (state, target_value) = self.replay_buffer.sample_batch(
                self.batch_size,
                sample_method="random_unique",
                concatenate=concatenate_samples,
                sample_attrs=["state", "value"],
                additional_concat_attrs=["value"],
            )

            if not concatenate_samples:
                target_value = t.Tensor(target_value).view(batch_size, 1)

            # calculate value loss
            value = self._criticize(state)
            value_loss = (
                self.criterion(target_value.type_as(value), value) * self.value_weight
            )

            if self.visualize:
                self.visualize_model(value_loss, "critic", self.visualize_dir)

            # Update critic network
            if update_value:
                self.critic.zero_grad()
                self._backward(value_loss)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_max)
                self.critic_optim.step()
            sum_value_loss += value_loss.item()

        self.replay_buffer.clear()
        self.actor.eval()
        self.critic.eval()
        return (
            act_policy_loss.item(),
            sum_value_loss / self.critic_update_times,
        )

    def _fvp_fim(self, state, vector: t.Tensor, damping: float):
        batch_size = len(state["state"])

        # M is the second derivative of the KL distance w.r.t. network output
        # (M*M diagonal matrix compressed into a M*1 vector)
        M, act_param = safe_call(self.actor, state, method="get_fim")

        # From now on we will use symbol `mu` as the action parameter of the
        # distribution, this symbol is used in equation 56. and 57. of the
        # paper
        mu = act_param.view(-1)

        # t is an arbitrary constant vector that does not depend on actor parameter
        # theta, we use t_ here since torch is imported as t
        t_ = t.ones(mu.shape, requires_grad=True, device=mu.device)
        mu_t = (mu * t_).sum()
        Jt = self.get_flat_grad(mu_t, list(self.actor.parameters()), create_graph=True)
        Jtv = (Jt * vector).sum()
        Jv = t.autograd.grad(Jtv, t_)[0]
        MJv = M * Jv.detach()
        mu_MJv = (MJv * mu).sum()
        JTMJv = self.get_flat_grad(mu_MJv, list(self.actor.parameters())).detach()
        JTMJv /= batch_size
        return JTMJv + vector * damping

    @staticmethod
    def _conjugate_gradients(Avp_f, b, eps, iterations, res_threshold):
        x = t.zeros(b.shape, dtype=b.dtype, device=b.device)
        r = b.clone()
        p = b.clone()
        r_dot_r = t.dot(r, r)
        best_x = None
        best_residual = np.inf
        patience = int(0.2 * iterations)
        for i in range(iterations):
            Avp = Avp_f(p)
            alpha = r_dot_r / (t.dot(p, Avp) + eps)
            x += alpha * p
            r -= alpha * Avp
            new_r_dot_r = t.dot(r, r)
            beta = new_r_dot_r / (r_dot_r + eps)
            p = r + beta * p
            if new_r_dot_r < best_residual:
                best_x = x.clone()
                best_residual = new_r_dot_r
            else:
                patience -= 1
            r_dot_r = new_r_dot_r
            if r_dot_r < res_threshold or patience == 0:
                break
        return best_x
