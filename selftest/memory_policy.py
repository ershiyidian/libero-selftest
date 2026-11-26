import collections

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prismatic.models.action_heads import L1RegressionActionHead, MLPResNetBlock_Pro
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class PhysicalTransitionMemory(nn.Module):
    def __init__(self, dim, slots=8, heads=8, action_horizon=8, visual_tokens=64):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.action_horizon = action_horizon
        self.visual_tokens = visual_tokens

        self.slot_identity = nn.Parameter(torch.empty(1, slots, dim))
        nn.init.normal_(self.slot_identity, std=0.02)
        self.visual_norm = nn.LayerNorm(dim)
        self.visual_projection = nn.Linear(dim, dim)
        self.visual_delta_projection = nn.Linear(dim, dim)
        self.proprio_projection = nn.Linear(8, dim)
        self.proprio_delta_projection = nn.Linear(8, dim)
        self.executed_action_projection = nn.Linear(action_horizon * ACTION_DIM + 1, dim)
        self.update_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.memory_norm = nn.LayerNorm(dim)
        self.update_norm = nn.LayerNorm(dim)
        self.update_ffn = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self,
        prior_memory,
        visual,
        previous_visual,
        proprio,
        previous_proprio,
        executed_actions,
        executed_count,
    ):
        if prior_memory.shape[-2:] != (self.slots, self.dim):
            raise ValueError("Bad recurrent-memory shape")
        if visual.shape != (prior_memory.shape[0], self.visual_tokens, self.dim):
            raise ValueError("Bad current visual embedding shape")
        if proprio.shape != (prior_memory.shape[0], 8):
            raise ValueError("LIBERO proprio must have exactly 8 dimensions")
        if executed_actions.shape != (prior_memory.shape[0], self.action_horizon, ACTION_DIM):
            raise ValueError("Bad actual-executed-actions history shape")

        visual = self.visual_norm(visual)
        delta_visual = visual - previous_visual
        delta_proprio = proprio - previous_proprio
        action_input = torch.cat([executed_actions.flatten(1), executed_count[:, None]], dim=-1)
        context = torch.cat(
            [
                self.visual_projection(visual),
                self.visual_delta_projection(delta_visual),
                self.proprio_projection(proprio).unsqueeze(1),
                self.proprio_delta_projection(delta_proprio).unsqueeze(1),
                self.executed_action_projection(action_input).unsqueeze(1),
            ],
            dim=1,
        )
        queries = prior_memory + self.slot_identity
        update, _ = self.update_attention(queries, context, context, need_weights=False)
        state = self.memory_norm(prior_memory + update)
        return self.update_norm(state + self.update_ffn(state)), visual


class IndependentMemoryRead(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, original_hidden, memory_tokens):
        memory_effect, _ = self.cross_attention(
            query=original_hidden, key=memory_tokens, value=memory_tokens, need_weights=False
        )
        return original_hidden + torch.tanh(self.gate) * memory_effect


class MemoryAugmentedProHead(L1RegressionActionHead):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        action_dim=7,
        slots=8,
        num_read_blocks=4,
        num_task_tokens=512,
    ):
        if action_dim != ACTION_DIM or NUM_ACTIONS_CHUNK != 8:
            raise ValueError("This implementation targets the official LIBERO 8-by-7 action interface")
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_task_tokens=num_task_tokens,
            use_pro_version=True,
        )
        self.slots = slots
        self.memory_update = PhysicalTransitionMemory(hidden_dim, slots=slots)
        blocks = self.model.mlp_resnet_blocks
        if len(blocks) < num_read_blocks or not all(isinstance(b, MLPResNetBlock_Pro) for b in blocks):
            raise ValueError("The loaded policy must have genuine Pro attention blocks")
        self.memory_readers = nn.ModuleList(
            [IndependentMemoryRead(hidden_dim, heads=8) for _ in range(num_read_blocks)]
        )
        self._active_memory = None
        self.memory_state = None
        self.previous_visual = None
        self.previous_proprio = None
        self._executed = collections.deque(maxlen=NUM_ACTIONS_CHUNK)
        self._hooks = []
        for index, block in enumerate(blocks[-num_read_blocks:]):
            self._hooks.append(block.register_forward_hook(self._make_read_hook(index)))

    def _make_read_hook(self, index):
        def memory_hook(_block, _inputs, original_output):
            if self._active_memory is None:
                return original_output
            return self.memory_readers[index](original_output, self._active_memory)

        return memory_hook

    def load_original_checkpoint(self, state_dict):
        normalized = {
            (name[7:] if name.startswith("module.") else name): tensor
            for name, tensor in state_dict.items()
        }
        result = self.load_state_dict(normalized, strict=False)
        expected_prefixes = ("memory_update.", "memory_readers.")
        invalid_missing = [name for name in result.missing_keys if not name.startswith(expected_prefixes)]
        if result.unexpected_keys or invalid_missing:
            raise RuntimeError(
                f"Pretrained Pro weights do not match original network: "
                f"missing={invalid_missing}, unexpected={result.unexpected_keys}"
            )
        if not result.missing_keys:
            raise RuntimeError("No new memory weights detected; the memory module was not registered")
        return len(normalized)

    def reset_memory(self):
        self.memory_state = None
        self.previous_visual = None
        self.previous_proprio = None
        self._active_memory = None
        self._executed.clear()

    def record_executed_action(self, physical_action):
        action = np.asarray(physical_action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Expected one real, finite, seven-dimensional control action")
        self._executed.append(action.copy())

    def _actual_actions_tensor(self, batch, device, dtype):
        history = torch.zeros(
            batch, NUM_ACTIONS_CHUNK, ACTION_DIM, device=device, dtype=dtype
        )
        if len(self._executed):
            if batch != 1:
                raise ValueError("A single environment's recorded actions cannot be broadcast to a batch")
            history[0, -len(self._executed):] = torch.as_tensor(
                np.stack(tuple(self._executed)), device=device, dtype=dtype
            )
        count = torch.full((batch,), len(self._executed) / NUM_ACTIONS_CHUNK, device=device, dtype=dtype)
        return history, count

    def _advance_memory(self, all_hidden_states, proprio):
        if all_hidden_states.ndim != 4:
            raise ValueError("Expected authentic multi-layer VLM features [B,L,T,D]")
        batch, _layers, tokens, width = all_hidden_states.shape
        if tokens <= self.num_task_tokens or width != self.hidden_dim:
            raise ValueError("VLM output and official action-head dimensions are inconsistent")
        visual = all_hidden_states[:, -1, :self.num_task_tokens]
        visual = F.adaptive_avg_pool1d(
            visual.transpose(1, 2), self.memory_update.visual_tokens
        ).transpose(1, 2)
        device, dtype = visual.device, visual.dtype
        robot_state = torch.as_tensor(proprio, dtype=dtype, device=device).reshape(batch, 8)
        previous_visual = (
            self.previous_visual if self.previous_visual is not None else torch.zeros_like(visual)
        )
        previous_proprio = (
            self.previous_proprio if self.previous_proprio is not None else torch.zeros_like(robot_state)
        )
        prior_memory = (
            self.memory_state
            if self.memory_state is not None
            else torch.zeros(batch, self.slots, self.hidden_dim, device=device, dtype=dtype)
        )
        # 尚未执行过任何物理动作时，记忆虽然可初始化，但不能干扰预训练动作。
        # 首次真实执行后才允许记忆残差参与Policy，这也对应“先试探，再利用结果”。
        has_executed_history = bool(self._executed)
        actions, count = self._actual_actions_tensor(batch, device, dtype)
        self.memory_state, self.previous_visual = self.memory_update(
            prior_memory,
            visual,
            previous_visual,
            robot_state,
            previous_proprio,
            actions,
            count,
        )
        self.previous_proprio = robot_state
        self._executed.clear()
        self._active_memory = self.memory_state if has_executed_history else None

    def predict_action(
        self,
        actions_hidden_states,
        proprio=None,
        proprio_projector=None,
        phase="Inference",
    ):
        if proprio is None or proprio_projector is None:
            raise ValueError("The memory-aware LIBERO head needs current robot state and its projector")
        self._advance_memory(actions_hidden_states, proprio)
        try:
            return super().predict_action(
                actions_hidden_states,
                proprio=proprio,
                proprio_projector=proprio_projector,
                phase=phase,
            )
        finally:
            self._active_memory = None
