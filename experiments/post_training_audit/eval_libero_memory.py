import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "VLA-Adapter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VENDOR))
os.chdir(VENDOR)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

LIBERO_ROOT = ROOT.parent / "external" / "LIBERO" / "libero" / "libero"
LIBERO_CONFIG = ROOT / "cache" / "libero_config"
LIBERO_CONFIG.mkdir(parents=True, exist_ok=True)
paths = {
    "benchmark_root": str(LIBERO_ROOT),
    "bddl_files": str(LIBERO_ROOT / "bddl_files"),
    "init_states": str(LIBERO_ROOT / "init_files"),
    "assets": str(LIBERO_ROOT / "assets"),
    "datasets": str(ROOT.parent / "cache" / "datasets"),
}
for label, location in paths.items():
    if not Path(location).is_dir():
        raise FileNotFoundError(f"LIBERO {label} path is unavailable: {location}")
(LIBERO_CONFIG / "config.yaml").write_text(yaml.safe_dump(paths))
os.environ["LIBERO_CONFIG_PATH"] = str(LIBERO_CONFIG)

from experiments.robot.libero.run_libero_eval import (
    TASK_MAX_STEPS,
    GenerateConfig,
    check_unnorm_key,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from libero.libero import benchmark
from selftest.memory_policy import MemoryAugmentedProHead
from selftest.video_recorder import LiberoDualViewRecorder


def write_jsonl(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_cfg(checkpoint):
    cfg = GenerateConfig(
        pretrained_checkpoint=str(checkpoint),
        task_suite_name="libero_spatial",
        use_pro_version=True,
        use_proprio=True,
        num_images_in_input=2,
        num_open_loop_steps=8,
        use_film=False,
        use_minivlm=True,
        center_crop=True,
    )
    return cfg


def load_policy(checkpoint):
    cfg = make_cfg(checkpoint)
    vlm, original_head, proprio_projector, _noisy, processor = initialize_model(cfg)
    check_unnorm_key(cfg, vlm)
    memory_head = MemoryAugmentedProHead(
        input_dim=vlm.llm_dim,
        hidden_dim=vlm.llm_dim,
        action_dim=7,
        num_task_tokens=original_head.num_task_tokens,
    ).to(device=next(original_head.parameters()).device, dtype=next(original_head.parameters()).dtype)
    loaded = memory_head.load_original_checkpoint(original_head.state_dict())
    vlm.eval()
    original_head.eval()
    memory_head.eval()
    proprio_projector.eval()
    print(
        "WEIGHT_CHECK",
        json.dumps(
            {
                "base_parameters_loaded": loaded,
                "original_head_params": sum(p.numel() for p in original_head.parameters()),
                "new_memory_params": sum(
                    p.numel() for name, p in memory_head.named_parameters()
                    if name.startswith(("memory_update.", "memory_readers."))
                ),
                "all_memory_gates_zero": all(
                    bool(torch.count_nonzero(reader.gate) == 0) for reader in memory_head.memory_readers
                ),
            }
        ),
        flush=True,
    )
    return cfg, vlm, original_head, memory_head, proprio_projector, processor


def get_chunk(cfg, vlm, obs, task_description, processor, head, projector):
    observation, image = prepare_observation(obs, get_image_resize_size(cfg))
    action = get_action(
        cfg=cfg,
        model=vlm,
        obs=observation,
        task_label=task_description,
        processor=processor,
        action_head=head,
        proprio_projector=projector,
        noisy_action_projector=None,
        use_film=False,
        use_minivlm=True,
    )
    chunk = np.asarray(action, dtype=np.float32)
    if chunk.shape != (8, 7) or not np.isfinite(chunk).all():
        raise ValueError(f"Official continuous Pro policy returned invalid action shape: {chunk.shape}")
    return chunk, image


def run_task(
    cfg,
    suite,
    task_id,
    episode_id,
    vlm,
    original_head,
    head,
    projector,
    processor,
    output_dir,
    verify_equivalence,
    save_video=True,
):
    task = suite.get_task(task_id)
    env, description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    t0 = time.perf_counter()
    actions_executed = []
    success = False
    equivalence = None
    recorder = None
    try:
        env.reset()
        initial_states = suite.get_task_init_states(task_id)
        if episode_id >= len(initial_states):
            raise ValueError(f"Episode index {episode_id} exceeds available saved task initial states")
        obs = env.set_init_state(initial_states[episode_id])
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

        if save_video:
            video_path = output_dir / f"task{task_id:02d}_episode{episode_id:02d}_two_views.mp4"
            recorder = LiberoDualViewRecorder(video_path)
            recorder.append(obs)

        head.reset_memory()
        queue = collections.deque()
        decision_index = 0
        for step in range(TASK_MAX_STEPS[cfg.task_suite_name]):
            if not queue:
                # 相同真实相机与本体，比较原模型和新增零门控模型的动作。
                if verify_equivalence and equivalence is None:
                    orig_chunk, _ = get_chunk(
                        cfg, vlm, obs, description, processor, original_head, projector
                    )
                    head.reset_memory()
                    updated_chunk, _ = get_chunk(
                        cfg, vlm, obs, description, processor, head, projector
                    )
                    equivalence = float(np.max(np.abs(orig_chunk - updated_chunk)))
                    print("ZERO_GATE_REAL_OBSERVATION_MAX_ABS", equivalence, flush=True)
                    if equivalence > 1e-5:
                        raise AssertionError("Memory injection altered pre-trained actions with zero gate")
                else:
                    updated_chunk, _ = get_chunk(
                        cfg, vlm, obs, description, processor, head, projector
                    )
                queue.extend(updated_chunk)
                decision_index += 1
            normalized_action = queue.popleft()
            physical_action = process_action(normalized_action, cfg.model_family)
            obs, reward, done, info = env.step(physical_action)
            head.record_executed_action(physical_action)
            actions_executed.append(np.asarray(physical_action, dtype=np.float32))
            if recorder is not None:
                recorder.append(obs)
            if done:
                success = True
                break
        result = {
            "suite": cfg.task_suite_name,
            "task_id": task_id,
            "episode": episode_id,
            "task": description,
            "success": bool(success),
            "executed_steps": len(actions_executed),
            "decisions": decision_index,
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "zero_gate_real_observation_max_abs": equivalence,
            "memory_slots": head.slots,
            "all_gates_zero": all(bool(torch.count_nonzero(b.gate) == 0) for b in head.memory_readers),
            "video_path": str(recorder.path) if recorder is not None else None,
            "video_frames": recorder.frames if recorder is not None else None,
        }
        np.savez_compressed(
            output_dir / f"task{task_id:02d}_episode{episode_id:02d}_actual_actions.npz",
            actions=np.asarray(actions_executed, dtype=np.float32),
            memory=head.memory_state.detach().float().cpu().numpy(),
        )
        return result
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "spatial_pro")
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "spatial_memory_zero_gate")
    parser.add_argument("--no-equivalence-check", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("Invalid requested task or episode list")
    # 官方模型导入时会切换到vendor工作目录；结果始终归档到本项目根目录。
    if not args.output.is_absolute():
        args.output = ROOT / args.output
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed_everywhere(7)
    cfg, vlm, original_head, head, projector, processor = load_policy(args.checkpoint.resolve())
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    if any(task < 0 or task >= suite.n_tasks for task in args.tasks):
        raise ValueError("At least one task index is outside the LIBERO suite")
    for task_id in args.tasks:
        for episode in range(args.episodes):
            row = run_task(
                cfg, suite, task_id, episode, vlm, original_head, head, projector,
                processor, args.output, not args.no_equivalence_check,
                save_video=not args.no_video,
            )
            write_jsonl(args.output / "episodes.jsonl", row)
            print("REAL_EPISODE", json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
