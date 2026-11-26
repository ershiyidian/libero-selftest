"""Clone robosuite operational-space-controller state alongside MuJoCo physics."""

import numpy as np
import torch


CONTROLLER_FIELDS = (
    "goal_pos", "goal_ori", "ori_ref", "relative_ori", "kp", "kd",
    "ee_pos", "ee_ori_mat", "ee_pos_vel", "ee_ori_vel",
)


def safe_snapshot(value):
    """Store trusted MuJoCo snapshots as tensor/scalar-only PyTorch data."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value.copy()))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: safe_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_snapshot(item) for item in value]
    if isinstance(value, (type(None), str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported snapshot value: {type(value)}")


def numpy_snapshot(value):
    """Decode weights_only=True snapshots for Robosuite and MuJoCo."""
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: numpy_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [numpy_snapshot(item) for item in value]
    return value
INTERPOLATOR_FIELDS = ("start", "goal", "step", "total_steps")


def _copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (float, int, np.floating, np.integer)):
        return value.item() if isinstance(value, np.generic) else value
    if value is None:
        return None
    raise TypeError(f"不能安全序列化控制器属性：{type(value)}")


def capture_controller(env):
    controller = env.robots[0].controller
    record = {
        "controller_class": type(controller).__name__,
        "fields": {}, "interpolators": {},
    }
    for name in CONTROLLER_FIELDS:
        if hasattr(controller, name):
            record["fields"][name] = _copy(getattr(controller, name))
    for name in ("interpolator_pos", "interpolator_ori"):
        interpolator = getattr(controller, name, None)
        if interpolator is not None:
            record["interpolators"][name] = {
                key: _copy(getattr(interpolator, key))
                for key in INTERPOLATOR_FIELDS if hasattr(interpolator, key)
            }
    return record


def restore_controller(env, snapshot):
    controller = env.robots[0].controller
    if type(controller).__name__ != snapshot["controller_class"]:
        raise ValueError("两个LIBERO环境的控制器类型不同")
    for name, value in snapshot["fields"].items():
        setattr(controller, name, _copy(value))
    for name, fields in snapshot["interpolators"].items():
        interpolator = getattr(controller, name)
        if interpolator is None:
            raise ValueError("恢复目标没有所需的插值器")
        for key, value in fields.items():
            setattr(interpolator, key, _copy(value))


def controller_difference(a, b):
    values = []
    for name, state in a["fields"].items():
        if state is not None:
            values.append(float(np.max(np.abs(np.asarray(state) - np.asarray(b["fields"][name])))))
    for name, fields in a["interpolators"].items():
        for key, value in fields.items():
            if value is not None:
                values.append(float(np.max(np.abs(np.asarray(value) - np.asarray(b["interpolators"][name][key])))))
    return max(values, default=0.0)


def replay_prefix(environment, snapshot, initial_state, *, num_steps_wait, no_op_action):
    """LIBERO控制器有隐藏状态，重放共同的物理前缀以重建完整执行上下文。"""
    environment.reset()
    observed = environment.set_init_state(initial_state)
    for _ in range(num_steps_wait):
        observed, _, _, _ = environment.step(no_op_action)
    for action in snapshot["prefix_actions"]:
        observed, _, done, _ = environment.step(action)
        if done:
            raise RuntimeError("共享前缀回放期间任务提前完成")
    return environment.regenerate_obs_from_state(snapshot["sim_state"])
