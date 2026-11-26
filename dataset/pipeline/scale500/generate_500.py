import argparse
import concurrent.futures as futures
import contextlib
import json
import multiprocessing
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from planning_500 import make_plan, load_catalog


HERE = Path(__file__).resolve().parent
LAB = HERE.parents[3]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def execute(plan: dict, run_directory: str) -> dict:
    # 每个进程只负责一个真实回合，错误返回批次管理器，其他回合继续执行。
    from engine_500 import Episode
    sys.path.insert(0, str(HERE.parent))
    from audit_dataset import audit
    root = Path(run_directory)
    path = root / "episodes" / plan["plan_id"]
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "plan.json", plan)
    began = time.monotonic()
    episode = None
    result = {"plan": plan, "episode": plan["plan_id"], "directory": str(path)}
    with (path / "execution.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                episode = Episode(plan)
                metadata = episode.run()
                # 保存到新子目录，防止既有数据被覆盖。
                actual = path / "record"
                episode.save(actual, metadata)
                report = audit(actual / "trajectory.hdf5", metadata)
                result.update({
                    "metadata": metadata, "audit": report,
                    "eligible": bool(report["eligible_for_success_sft"]),
                    "simulation_completed": True, "error": None,
                })
            except Exception as error:
                log.write(traceback.format_exc())
                result.update({
                    "eligible": False, "simulation_completed": False,
                    "error": f"{type(error).__name__}: {error}",
                })
                if episode is not None and len(episode.records) == len(episode.actions) + 1:
                    partial = {
                        "plan": plan, "success": False, "invalid_physics": True,
                        "events": episode.events, "error": result["error"],
                    }
                    episode.save(path / "invalid_record", partial)
            finally:
                if episode is not None:
                    episode.close()
    result["elapsed_seconds"] = round(time.monotonic() - began, 2)
    write_json(path / "outcome.json", result)
    return result


def inspect_sources() -> dict:
    import h5py
    from engine_500 import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    result = {}
    for task in range(10):
        name = suite.get_task(task).name
        file = LAB / "cache/datasets/libero/libero_spatial" / (name + "_demo.hdf5")
        with h5py.File(file) as source:
            demos = [
                int(key.rsplit("_", 1)[-1]) for key, item in source["data"].items()
                if len(item["actions"]) >= 40 and item["rewards"][-1] > 0
            ]
        assert demos
        result[task] = {"task": name, "demos": sorted(demos), "source": str(file)}
    return result


def status(root, rows, selected, target, started, assigned, state):
    accepted = [row for row in rows if row["episode"] in selected]
    values = Counter()
    methods = Counter()
    conditions = Counter()
    splits = Counter()
    unique = set()
    recovered = 0
    for row in accepted:
        p = row["plan"]
        values[str(p["task_id"])] += 1
        methods[p["first_method"]] += 1
        conditions.update(p["condition_ids"])
        unique.add(p["split_group"])
        splits[p["split"]] += 1
        recovered += row["audit"]["negative_tests"] > 0
    progress = {
        "run_name": root.name, "target_successes": target,
        "producer_pid": os.getpid(),
        "accepted_successes": len(selected), "attempts_completed": len(rows),
        "attempts_assigned": assigned,
        "physics_completed": sum(row["simulation_completed"] for row in rows),
        "task_failures": sum(row["simulation_completed"] and not row["eligible"] for row in rows),
        "execution_errors": sum(not row["simulation_completed"] for row in rows),
        "successful_recoveries": recovered,
        "task_counts": dict(values), "first_method_counts": dict(methods),
        "condition_counts": dict(conditions), "split_counts": dict(splits),
        "unique_expert_histories": len(unique),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": state, "generation_finished": state != "running",
        "target_reached": len(selected) >= target,
        "rendering": "CPU software rendering", "model_weights_loaded": False,
    }
    write_json(root / "progress.json", progress)
    entries = [{
        "episode": row["episode"],
        "trajectory": f"episodes/{row['episode']}/record/trajectory.hdf5",
        "language": row["metadata"]["language"],
        "split_group": row["plan"]["split_group"],
        "split": row["plan"]["split"],
        "source_demo": row["plan"]["demo_id"],
    } for row in accepted]
    write_json(root / "training_index.json", {
        "episodes": entries, "hidden_truth_excluded": True,
        "policy_inputs": ["images", "proprio", "actions", "history"],
    })
    write_json(root / "manifest.json", {"progress": progress, "selected": sorted(selected), "records": rows})
    return progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.workers <= 8 and args.target >= 10
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    (root / "episodes").mkdir(exist_ok=True)
    (root / "producer.pid").write_text(str(os.getpid()))
    catalog = load_catalog()
    write_json(root / "run_config.json", {
        "target_successes": args.target, "workers": args.workers,
        "max_attempts": args.max_attempts, "seed": args.seed,
        "task_ids": list(range(10)), "conditions": ["C01", "C02", "C03", "C04"],
        "methods": ["P01", "P02", "P03", "P04", "P05", "P07"],
        "software_rendering": os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
        "base_model_training": False,
    })
    sources = inspect_sources()
    write_json(root / "source_inventory.json", sources)
    rows = []
    selected = set()
    previous_elapsed = 0.0
    if args.resume and (root / "manifest.json").exists():
        prior = json.loads((root / "manifest.json").read_text())
        rows, selected = prior["records"], set(prior["selected"])
        previous_elapsed = float(prior["progress"]["elapsed_seconds"])
    completed_ids = {row["episode"] for row in rows}
    for file in sorted((root / "episodes").glob("*/outcome.json")):
        outcome = json.loads(file.read_text())
        if outcome["episode"] not in completed_ids:
            rows.append(outcome)
            completed_ids.add(outcome["episode"])
    quotas = {task: args.target // 10 + int(task < args.target % 10) for task in range(10)}
    accepted_by_task = Counter(row["plan"]["task_id"] for row in rows if row["episode"] in selected)
    source_used = Counter(row["plan"]["split_group"] for row in rows if row["episode"] in selected)
    submitted_by_task = Counter(row["plan"]["task_id"] for row in rows)
    existing_numbers = [int(path.name.rsplit("_", 1)[-1]) for path in (root / "episodes").glob("episode_*")]
    assigned = max(existing_numbers, default=-1) + 1
    sample_rng = random.Random(args.seed + assigned)
    public_demo_order = {}
    for task, source in sources.items():
        demo_ids = source["demos"].copy()
        random.Random(args.seed + task).shuffle(demo_ids)
        public_demo_order[task] = demo_ids
    started = time.monotonic() - previous_elapsed
    status(root, rows, selected, args.target, started, assigned, "running")
    process_context = multiprocessing.get_context("spawn")
    stopped = False
    with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=process_context) as executor:
        pending = {}
        while True:
            pending_tasks = Counter(plan["task_id"] for plan in pending.values())
            candidates = [
                task for task in range(10)
                if accepted_by_task[task] + pending_tasks[task] < quotas[task]
            ]
            while len(pending) < args.workers and assigned < args.max_attempts and candidates and not (root/"STOP").exists():
                priority = min(accepted_by_task[t] + pending_tasks[t] for t in candidates)
                task = sample_rng.choice([t for t in candidates if accepted_by_task[t] + pending_tasks[t] == priority])
                order = public_demo_order[task]
                offset = submitted_by_task[task] % len(order)
                demo = order[offset]
                # 同一个原始示范最多进入三个成功配置，扩大真实操作历史覆盖。
                for rotation in range(len(order)):
                    candidate_demo = order[(offset + rotation) % len(order)]
                    if source_used[f"spatial_{task}_demo_{candidate_demo}"] < 3:
                        demo = candidate_demo
                        break
                plan = make_plan(task, demo, assigned, args.seed, catalog)
                job = executor.submit(execute, plan, str(root))
                pending[job] = plan
                assigned += 1
                submitted_by_task[task] += 1
                pending_tasks[task] += 1
                candidates = [t for t in range(10) if accepted_by_task[t] + pending_tasks[t] < quotas[t]]
            if not pending:
                break
            done, _ = futures.wait(pending, timeout=15, return_when=futures.FIRST_COMPLETED)
            for job in done:
                plan = pending.pop(job)
                outcome = job.result()
                rows.append(outcome)
                task = plan["task_id"]
                if outcome["eligible"] and accepted_by_task[task] < quotas[task] and len(selected) < args.target:
                    selected.add(outcome["episode"])
                    accepted_by_task[task] += 1
                    source_used[plan["split_group"]] += 1
                print("PROGRESS", len(selected), "/", args.target, "attempts", len(rows), "task", task, "valid", outcome["eligible"], flush=True)
            status(root, rows, selected, args.target, started, assigned, "running")
            if (root / "STOP").exists():
                stopped = True
            if stopped and not pending:
                break
    final_state = "complete" if len(selected) >= args.target else "stopped" if stopped else "attempt_limit"
    result = status(root, rows, selected, args.target, started, assigned, final_state)
    if selected:
        sys.path.insert(0, str(HERE.parent))
        from audit_dataset import SelfTestEpisodes
        dataset = SelfTestEpisodes(root / "training_index.json")
        sample = dataset[0]
        assert sample["actions"].shape[-2:] == (8, 7)
        write_json(root / "loader_check.json", {
            "actual_episode_read": True,
            "shape": {key: list(value.shape) for key, value in sample.items() if hasattr(value, "shape")},
        })
    print("GENERATION_FINISHED", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
