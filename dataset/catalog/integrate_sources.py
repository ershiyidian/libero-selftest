import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def save(name, value):
    (SOURCES / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    tasks = json.loads((SOURCES / "task_registry.json").read_text())
    originals = json.loads((SOURCES / "original_task_bindings.json").read_text())
    by_stem = {Path(item["source"]).stem: item for item in originals}
    by_language = defaultdict(list)
    for item in originals:
        by_language[item["language"]].append(item)
    by_task = {row["source_task"]: row for row in tasks}
    for row in tasks:
        row["recovery_count"] = row["expert_count"]
        row["rlds_count"] = 0
        row["original_hdf5_count"] = 0

    def get_task(binding):
        name = binding["language"]
        if name not in by_task:
            value = {
                "task_id": f"T{len(tasks) + 1:03d}", "source_task": name,
                "source_task_indices": [], "episode_ids": [],
                "expert_count": 0, "frames": 0, "recovery_count": 0,
                "rlds_count": 0, "original_hdf5_count": 0,
                "registry": "additional_source_routes.jsonl",
            }
            by_task[name] = value
            tasks.append(value)
        return by_task[name]

    routes = []
    for item in rows(SOURCES / "rlds_episode_sources.jsonl"):
        key = "episode_metadata/file_path"
        source_path = item["original_episode_metadata"][key][0]
        stem = Path(source_path).stem.removesuffix("_demo")
        assert stem in by_stem, stem
        binding = by_stem[stem]
        task = get_task(binding)
        task["rlds_count"] += 1
        if not task["recovery_count"]:
            task["expert_count"] += 1
            task["frames"] += item["steps_from_language"]
        routes.append({
            "source_kind": "rlds", "task_id": task["task_id"],
            "source_task": binding["language"], "stored_language": item["languages"],
            "source_goal": binding["goal_predicates"],
            "source_bddl": binding["source"], "source_shard": item["shard"],
            "record_index": item["record_index"], "frame_count": item["steps_from_language"],
            "replay_binding": "ordinary_task_reference_requires_corresponding_simulator_state",
        })
    for item in json.loads((SOURCES / "original_hdf5_sources.json").read_text()):
        stem = Path(item["task_file"]).stem.removesuffix("_demo")
        assert stem in by_stem, stem
        binding = by_stem[stem]
        task = get_task(binding)
        task["original_hdf5_count"] += 1
        routes.append({
            "source_kind": "original_hdf5", "task_id": task["task_id"],
            "source_task": binding["language"], "source_bddl": binding["source"],
            "source_hdf5": item["task_file"], "source_demo": item["demo"],
            "frame_count": item["frames"], "has_full_states": item["full_states"],
            "source_goal": binding["goal_predicates"],
            "replay_binding": "full_expert_prefix_available",
        })
    with (SOURCES / "additional_source_routes.jsonl").open("w", encoding="utf-8") as output:
        for row in routes:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    save("all_task_registry.json", tasks)
    save("supplement_task_registry.json", tasks[78:])
    bindings = [{
        "task_id": item["task_id"], "source_task": item["source_task"],
        "exact_language_matches": by_language[item["source_task"]],
        "requires_alias_review": False,
    } for item in tasks]
    save("all_task_bindings.json", bindings)
    save("source_route_summary.json", {
        "recovery_tasks": 78, "all_tasks_with_available_demonstrations": len(tasks),
        "supplementary_native_tasks": len(tasks)-78,
        "rlds_records_routed": sum(x["source_kind"] == "rlds" for x in routes),
        "hdf5_records_routed": sum(x["source_kind"] == "original_hdf5" for x in routes),
        "all_recovery_records_in_separate_registry": 3184,
        "native_and_recovery_examples_may_overlap": True,
    })
    print("ALL_SOURCE_ROUTE_COUNTS", len(tasks), len(routes), flush=True)
    print("SUPPLEMENT_TASKS", json.dumps(tasks[78:], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
