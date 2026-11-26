import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/guest/vla_branch_lab/cache/envs/qwen35_joint/bin/python")
START = ROOT / "results/formal_native35_ordinary130_500labels/joint"
DEFAULT_OUTPUT = ROOT / "results/ordinary130_grpo65x16_lang05_five"
TASKS = [str(x) for x in range(0, 130, 2)]


def atomic_json(path, value):
    temp = path.with_suffix(".json.writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def step_checkpoint(path):
    return (path / "policy.json").is_file() and all(
        (path / name).is_file() and (path / name).stat().st_size > 1000
        for name in ("weights.pt", "rl_optimizer.pt")
    )


def execute(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    print("GRPO_EXECUTE", " ".join(map(str, args)), flush=True)
    with log.open("a", buffering=1) as handle:
        handle.write("GRPO_EXECUTE " + " ".join(map(str, args)) + "\n")
        subprocess.run(list(map(str, args)), cwd=ROOT, env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def evaluate_finished(path, macro, parent):
    roll = path / "rollouts"
    result = json.loads((roll / "rollout_report.json").read_text())
    if result["total"] != 1040 or result["done"] != 1040 or not result["ready_to_train"]:
        raise RuntimeError(f"第{macro}大步采样不完整：{result['done']}/1040")
    train = json.loads((path / "train_report.json").read_text())
    if train.get("status") not in ("passed", "no_reward_variance"):
        raise RuntimeError(f"第{macro}大步更新结果异常：{train.get('status')}")
    if train["status"] == "passed" and not step_checkpoint(path / "checkpoint"):
        raise RuntimeError(f"第{macro}大步模型权重或优化器未保存完毕")
    if train["status"] == "no_reward_variance" and step_checkpoint(path / "checkpoint"):
        raise RuntimeError("无奖励差异却存在新检查点")
    groups = result["groups"]
    successes = sum(row["successes"] for row in groups.values())
    mixed = sum(0 < row["successes"] < 16 for row in groups.values())
    next_checkpoint = path / "checkpoint" if train["status"] == "passed" else parent
    return {
        "macro_step": macro,
        "episodes": result["done"], "tasks": len(groups), "trials_per_task": 16,
        "total_successes": successes, "groups_with_mixed_rewards": mixed,
        "groups_with_all_failures": sum(row["successes"] == 0 for row in groups.values()),
        "rollout_seconds": result["elapsed_seconds"],
        "training_status": train["status"],
        "active_advantage_groups": train.get("active_advantage_groups", 0),
        "actual_trained_action_chunks": train.get("actual_trained_action_chunks", 0),
        "next_checkpoint": str(next_checkpoint.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-after", type=int, default=5)
    parser.add_argument("--max-control-steps", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.stop_after <= 5:
        raise ValueError("本次只安排五个GRPO大步")
    if args.max_control_steps < 8:
        raise ValueError("控制步上限至少覆盖一个8步动作块")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(TASKS) != 65:
        raise AssertionError("原65任务列表长度不正确")
    finished = []
    parent = START.resolve()
    for macro in range(1, args.stop_after + 1):
        path = output / f"step{macro:02d}"
        path.mkdir(parents=True, exist_ok=True)
        rollouts = path / "rollouts"
        report = path / "train_report.json"
        checkpoint = path / "checkpoint"
        if report.is_file():
            item = evaluate_finished(path, macro, parent)
            finished.append(item)
            atomic_json(output / "macro_progress.json", finished)
            parent = Path(item["next_checkpoint"])
            print("GRPO_RESTORED_MACRO", json.dumps(item), flush=True)
            continue
        atomic_json(output / "run_state.json", {
            "phase": "rollouts", "current_macro": macro,
            "total_macro": args.stop_after, "completed": len(finished),
            "start_checkpoint": str(START), "parent_checkpoint": str(parent),
        })
        if not (rollouts / "rollout_report.json").is_file():
            command = [
                PYTHON, "-u", ROOT / "singlepass_vla/launch_grpo.py",
                "--catalog", "libero130", "--tasks", *TASKS,
                "--gpus", "0", "1", "2", "--workers-per-gpu", "2",
                "--trials-per-task", "16",
                "--max-control-steps", str(args.max_control_steps),
                "--checkpoint", parent, "--language-tokens", "32",
                "--language-temperature", "0.5", "--action-temperature", "1.0",
                "--sigma", "0.1", "--rho", "0.8",
                "--scenario-source", "official", "--reward-mode", "official_success",
                "--seed-offset", str((macro - 1) * 1000000),
                "--output", rollouts,
            ]
            if (rollouts / "rollout_queue.sqlite3").exists():
                command.append("--resume")
            execute(command, path / "rollouts.log")
        specs = json.loads((rollouts / "rollout_report.json").read_text())
        current = Path(specs["spec"]["checkpoint"]).resolve()
        atomic_json(output / "run_state.json", {
            "phase": "training", "current_macro": macro,
            "total_macro": args.stop_after, "completed": len(finished),
            "source_checkpoint": str(current),
        })
        execute([
            PYTHON, "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc-per-node=3",
            ROOT / "benchmarks/native_qwen35_grpo_macro.py",
            "--rollouts", rollouts, "--checkpoint", current,
            "--trials-per-task", "16", "--lr", "0.000001",
            "--clip-epsilon", "0.2", "--sft-replay-weight", "0.02",
            "--general-replay-weight", "0.005",
            "--report", report, "--save-checkpoint", checkpoint,
            "--save-optimizer",
        ], path / "train.log")
        item = evaluate_finished(path, macro, parent)
        finished.append(item)
        atomic_json(output / "macro_progress.json", finished)
        parent = Path(item["next_checkpoint"])
        print("GRPO_MACRO_RESULT", json.dumps(item, ensure_ascii=False), flush=True)
    atomic_json(output / "run_state.json", {
        "phase": "completed", "completed": len(finished),
        "total_macro": args.stop_after, "last_checkpoint": str(parent),
    })
    print("GRPO_FIVE_MACROS_FINISHED", json.dumps(finished, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
