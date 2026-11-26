import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogue", type=Path, default=ROOT / "deliverable")
    args = parser.parse_args()
    folder = args.catalogue
    cards = json.loads((folder / "catalogue.json").read_text(encoding="utf-8"))["tasks"]
    by_id = {card["task_id"]: card for card in cards}
    sources = read_rows(folder / "source_reuse_plan.jsonl")
    grouped = defaultdict(list)
    scenes = defaultdict(list)
    for source in sources:
        if source["role"] == "scene_observation":
            scenes[source["task_id"]].append(source["source_id"])
        else:
            grouped[source["task_id"]].append(source)
    # 同一轮遍历各任务的来源，保留少样本任务的覆盖机会。
    order = []
    for index in range(max(len(items) for items in grouped.values())):
        for task_id in sorted(grouped):
            if index < len(grouped[task_id]):
                order.append(grouped[task_id][index])
    counts = Counter()
    source_seen = set()
    total = 0
    with (folder / "source_method_schedule.jsonl").open("w", encoding="utf-8") as output:
        for method_number in range(10):
            for source in order:
                card = by_id[source["task_id"]]
                method = card["methods"][method_number]
                row = {
                    "plan_id": f"coverage_{total:06d}",
                    "task_id": card["task_id"],
                    "source_id": source["source_id"],
                    "source_kind": source["source_kind"],
                    "method_id": method["id"],
                    "candidate_condition_ids": method["compatible_conditions"],
                    "source_locator": "source_reuse_plan.jsonl",
                    "status": "requires_binding_and_real_execution",
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[card["task_id"]] += 1
                source_seen.add(source["source_id"])
                total += 1
    summary = {
        "task_entries": len(grouped),
        "dynamic_source_records": len(order),
        "static_scene_reference_records": sum(map(len, scenes.values())),
        "source_method_plans": total,
        "each_dynamic_source_method_count": 10,
        "covered_source_records": len(source_seen),
        "unassigned_source_records": 0,
        "per_task_plan_counts": dict(counts),
        "simulation_rollouts": 0,
    }
    assert len(source_seen) == len(order)
    assert len(grouped) == len(cards)
    assert total == len(order) * 10
    (folder / "source_method_schedule_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (folder / "static_scene_references.json").write_text(
        json.dumps(dict(scenes), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SOURCE_METHOD_COVERAGE", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
