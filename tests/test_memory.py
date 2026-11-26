import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "VLA-Adapter"))
os.environ.setdefault("MUJOCO_GL", "egl")

from prismatic.models.action_heads import L1RegressionActionHead
from selftest.memory_policy import MemoryAugmentedProHead


def test_exact_original_output_with_zero_gate():
    torch.manual_seed(23)
    width = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = L1RegressionActionHead(
        input_dim=width, hidden_dim=width, action_dim=7,
        num_task_tokens=3, use_pro_version=True,
    ).to(device=device, dtype=torch.bfloat16).eval()
    augmented = MemoryAugmentedProHead(
        input_dim=width, hidden_dim=width, num_task_tokens=3,
        slots=8, num_read_blocks=4,
    ).to(device=device, dtype=torch.bfloat16).eval()
    assert augmented.load_original_checkpoint(base.state_dict()) == len(base.state_dict())
    hidden = torch.randn(1, 25, 3 + 64, width, device=device, dtype=torch.bfloat16)
    proprio = torch.tensor([.1, .2, -.3, .05, .12, .33, .8, .9], device=device, dtype=torch.bfloat16)
    projector = nn.Linear(8, width).to(device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        original = base.predict_action(hidden, proprio, projector)
        first = augmented.predict_action(hidden, proprio, projector)
        assert torch.equal(original, first), "Zero gate must retain exact original prediction"
        assert augmented.memory_state.shape == (1, 8, width)
        augmented.record_executed_action(np.array([0, .01, .01, 0, 0, 0, -1], dtype=np.float32))
        later = augmented.predict_action(hidden, proprio, projector)
        assert torch.equal(original, later), "Memory updates with zero gate must retain original action"
        assert torch.isfinite(augmented.memory_state).all()
    augmented.reset_memory()
    assert augmented.memory_state is None
    assert len(augmented._executed) == 0


def test_memory_is_trainable_across_decision_steps():
    torch.manual_seed(8)
    width = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    augmented = MemoryAugmentedProHead(
        input_dim=width, hidden_dim=width, num_task_tokens=3,
        slots=8, num_read_blocks=4,
    ).to(device=device, dtype=torch.bfloat16).train()
    projector = nn.Linear(8, width).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for reader in augmented.memory_readers:
            reader.gate.fill_(0.2)
    first_hidden = torch.randn(1, 25, 3 + 64, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    next_hidden = torch.randn(1, 25, 3 + 64, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    proprio = torch.ones(8, device=device, dtype=torch.bfloat16)
    augmented.predict_action(first_hidden, proprio, projector)
    augmented.record_executed_action(np.array([.0, 0.02, 0, 0, 0, 0, 1], dtype=np.float32))
    later = augmented.predict_action(next_hidden, proprio * 1.01, projector)
    loss = later.square().mean()
    loss.backward()
    assert first_hidden.grad is not None
    assert torch.isfinite(first_hidden.grad).all()
    assert first_hidden.grad.abs().sum().item() > 0
    assert augmented.memory_update.visual_projection.weight.grad is not None
    assert augmented.memory_update.visual_projection.weight.grad.abs().sum().item() > 0
    assert augmented.memory_readers[0].gate.grad is not None


def test_control_action_record_validation():
    head = MemoryAugmentedProHead(input_dim=32, hidden_dim=32, num_task_tokens=3)
    try:
        head.record_executed_action(np.zeros((8, 7), dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Only single physically executed control steps should be accepted")


def test_real_action_history_changes_recurrent_memory():
    torch.manual_seed(62)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MemoryAugmentedProHead(
        input_dim=32, hidden_dim=32, num_task_tokens=3,
        slots=8, num_read_blocks=4,
    ).to(device=device, dtype=torch.bfloat16).eval()
    projector = nn.Linear(8, 32).to(device=device, dtype=torch.bfloat16)
    observation_features = torch.randn(
        1, 25, 3 + 64, 32, device=device, dtype=torch.bfloat16
    )
    robot_state = torch.ones(8, device=device, dtype=torch.bfloat16)

    def rollout_with_action(actual_action):
        model.reset_memory()
        with torch.inference_mode():
            _ = model.predict_action(observation_features, robot_state, projector)
            model.record_executed_action(actual_action)
            prediction = model.predict_action(observation_features, robot_state, projector)
            return prediction.clone(), model.memory_state.clone()

    stable_action = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    lateral_action = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    prediction_a, memory_a = rollout_with_action(stable_action)
    prediction_b, memory_b = rollout_with_action(lateral_action)
    assert torch.equal(prediction_a, prediction_b), "Zero gate must protect identical observations"
    assert not torch.equal(memory_a, memory_b), "The memory must incorporate the executed physical action"


def test_nonzero_memory_gate_waits_for_actual_executed_history():
    torch.manual_seed(104)
    width = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = L1RegressionActionHead(
        input_dim=width, hidden_dim=width, action_dim=7,
        num_task_tokens=3, use_pro_version=True,
    ).to(device=device, dtype=torch.bfloat16).eval()
    memory = MemoryAugmentedProHead(
        input_dim=width, hidden_dim=width, num_task_tokens=3,
    ).to(device=device, dtype=torch.bfloat16).eval()
    memory.load_original_checkpoint(base.state_dict())
    projector = nn.Linear(8, width).to(device=device, dtype=torch.bfloat16).eval()
    hidden = torch.randn(1, 25, 3 + 64, width, device=device, dtype=torch.bfloat16)
    state = torch.randn(8, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        for reader in memory.memory_readers:
            reader.gate.fill_(0.5)
        standard = base.predict_action(hidden, state, projector)
        before_executed = memory.predict_action(hidden, state, projector)
        assert torch.equal(standard, before_executed), "No actual probe history: keep pretrained first action"
        memory.record_executed_action(
            np.array([0.01, 0.0, 0.01, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        )
        after_executed = memory.predict_action(hidden, state, projector)
        assert not torch.equal(standard, after_executed), "Actual history must enable memory-read residual"
