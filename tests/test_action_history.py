from pathlib import Path

import numpy as np
import torch

from prismatic.models.action_heads import L1RegressionActionHead
from selftest.action_history_policy import ActionHistoryProHead


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "official_training/runs/"
    "memory_pro_libero_spatial_no_noops--30000_chkpt/"
    "action_head--30000_checkpoint.pt"
)


def test_actual_trained_pro_exact_initial_equivalence_and_history_gradient():
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    loaded = torch.load(SOURCE, map_location="cpu", weights_only=True)
    parameters = {k.removeprefix("module."): v for k, v in loaded.items()}
    original = L1RegressionActionHead(
        input_dim=896, hidden_dim=896, action_dim=7, use_pro_version=True
    )
    original.load_state_dict(
        {k: v for k, v in parameters.items() if k.startswith("model.")}, strict=True
    )
    history = ActionHistoryProHead(
        input_dim=896, hidden_dim=896, action_dim=7, use_pro_version=True
    )
    assert history.load_old_head(loaded) == len(original.state_dict())
    device = torch.device("cuda")
    original = original.to(device=device, dtype=torch.bfloat16).eval()
    history = history.to(device=device, dtype=torch.bfloat16).eval()

    torch.manual_seed(20261011)
    features = torch.randn((1, 25, 520, 896), device=device, dtype=torch.bfloat16)
    proprio = torch.randn((1, 8), device=device, dtype=torch.bfloat16)
    projector = torch.nn.Linear(8, 896).to(device=device, dtype=torch.bfloat16)
    real_actions = np.load(
        ROOT / "results/official_memory_from_foundation_spatial_5000_no_memory/"
        "task00_episode00_physical.npz", allow_pickle=False
    )["actions"]
    for action in real_actions[:8]:
        history.record_executed_action(action)

    with torch.no_grad():
        target = original.predict_action(features, proprio, projector, phase="Inference")
        actual = history.predict_action(features, proprio, projector, phase="Inference")
    assert torch.equal(target, actual), "零初始化历史投影必须严格继承原版动作输出"
    assert history.history_projection.weight.count_nonzero().item() == 0

    # 使用真实LIBERO执行轨迹的历史动作来验证端到端梯度。
    history.train()
    history.history_projection.weight.requires_grad_(True)
    predicted = history.predict_action(features, proprio, projector, phase="Inference")
    loss = predicted.float().square().mean()
    loss.backward()
    grad = history.history_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().max() > 0

    history.reset_history()
    assert np.array_equal(history.get_executed_history(), np.zeros((8, 7), dtype=np.float32))
    for action in real_actions[:9]:
        history.record_executed_action(action)
    np.testing.assert_array_equal(history.get_executed_history(), real_actions[1:9])
