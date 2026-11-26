import collections

import numpy as np
import torch
from torch import nn

from prismatic.models.action_heads import L1RegressionActionHead, learnable_random_perturbations
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


class ActionHistoryProHead(L1RegressionActionHead):
    """用最近实际执行的动作替换Pro策略的八个零输入槽位。"""

    def __init__(self, input_dim, hidden_dim, action_dim=7, use_pro_version=True, num_task_tokens=512):
        if not use_pro_version or action_dim != ACTION_DIM or NUM_ACTIONS_CHUNK != 8:
            raise ValueError("历史动作策略必须使用8×7的VLA-Adapter-Pro")
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_task_tokens=num_task_tokens,
            use_pro_version=True,
        )
        # 零初始化保证加载旧动作网络时，第一次预测与原版完全一致。
        # 归一化层会放大非常小的输入，因此把历史动作加入原有输入层之后。
        # 原始24层策略及其全部预训练参数保持不变。
        self.history_projection = nn.Linear(action_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.history_projection.weight)
        self._executed = collections.deque(maxlen=NUM_ACTIONS_CHUNK)

    def reset_history(self):
        self._executed.clear()

    def record_executed_action(self, action):
        physical = np.asarray(action, dtype=np.float32)
        if physical.shape != (ACTION_DIM,) or not np.isfinite(physical).all():
            raise ValueError("实际执行动作必须是有限的7维向量")
        self._executed.append(physical.copy())

    def get_executed_history(self):
        history = np.zeros((NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=np.float32)
        if self._executed:
            history[-len(self._executed):] = np.stack(tuple(self._executed))
        return history

    def load_old_head(self, state_dict):
        # 允许从此前的8槽循环记忆实验继承全部原始Pro网络权重。
        normalized = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
            if not key.removeprefix("module.").startswith(("memory_update.", "memory_readers."))
        }
        loaded = self.load_state_dict(normalized, strict=False)
        if set(loaded.missing_keys) != {"history_projection.weight"} or loaded.unexpected_keys:
            raise RuntimeError(f"Pro权重不匹配：{loaded}")
        return len(normalized)

    def _history_tensor(self, batch, device, dtype, previous_actions, phase):
        if previous_actions is None:
            if phase == "Training":
                raise ValueError("SFT必须显式提供真实过去动作，避免历史标签缺失")
            if batch != 1:
                raise ValueError("并行环境必须分别传入各自的真实历史动作")
            previous_actions = self.get_executed_history()[None]
        history = torch.as_tensor(previous_actions, device=device, dtype=dtype)
        if history.shape != (batch, NUM_ACTIONS_CHUNK, ACTION_DIM):
            raise ValueError(f"历史动作应为[B,8,7]，收到{tuple(history.shape)}")
        if not torch.isfinite(history).all():
            raise ValueError("历史动作含非法数值")
        return history

    def predict_action(
        self,
        actions_hidden_states,
        proprio=None,
        proprio_projector=None,
        phase="Inference",
        previous_actions=None,
    ):
        if proprio is None or proprio_projector is None:
            raise ValueError("必须提供当前本体状态及其官方投影器")
        batch = actions_hidden_states.shape[0]
        dtype = actions_hidden_states.dtype
        history = self._history_tensor(batch, actions_hidden_states.device, dtype, previous_actions, phase)
        proprio_features = proprio_projector(proprio.reshape(batch, -1).to(dtype)).unsqueeze(1)
        task = actions_hidden_states[:, :, :self.num_task_tokens, :]
        action_features = actions_hidden_states[:, :, self.num_task_tokens:, :]

        # 保留原有8个空槽位及其预训练输入层。
        inputs = torch.zeros(
            batch, NUM_ACTIONS_CHUNK, ACTION_DIM * self.hidden_dim,
            device=history.device, dtype=history.dtype,
        )
        if phase == "Training":
            # 与作者的原动作头训练路径保持一致。
            inputs = inputs + learnable_random_perturbations(
                inputs.shape[1], inputs.shape[2], device=inputs.device, dtype=inputs.dtype
            )
        hidden = self.model.relu(self.model.fc1(self.model.layer_norm1(inputs)))
        hidden = hidden + self.history_projection(history)
        for index, block in enumerate(self.model.mlp_resnet_blocks):
            hidden = block(
                hidden,
                h_t=task[:, index + 1, :],
                h_a=action_features[:, index + 1, :],
                p=proprio_features,
            )
        return self.model.fc2(self.model.layer_norm2(hidden))

    def forward(self, actions_hidden_states, proprio, proprio_projector, previous_actions, phase="Training"):
        return self.predict_action(
            actions_hidden_states, proprio, proprio_projector, phase, previous_actions
        )
