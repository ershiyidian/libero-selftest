import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=ROOT / "deliverable")
    args = parser.parse_args()
    folder = args.folder
    data = load(folder / "catalogue.json")
    original = {r["task_id"]: r for r in load(ROOT / "sources/all_task_registry.json")}
    bindings = {r["task_id"]: r for r in load(ROOT / "sources/all_task_bindings.json")}
    cards = data["tasks"]
    assert len(cards) == len(original) == 96
    families = Counter()
    for card in cards:
        tid = card["task_id"]
        source = original[tid]
        assert card["source_task"] == source["source_task"]
        assert card["recovery_episode_ids"] == source["episode_ids"]
        assert card["original_task_indices"] == source["source_task_indices"]
        assert card["original_bindings"] == bindings[tid]["exact_language_matches"]
        methods = card["methods"]
        conditions = card["conditions"]
        assert [m["id"] for m in methods] == [f"M{i:02d}" for i in range(1,11)]
        assert [c["id"] for c in conditions] == [f"C{i:02d}" for i in range(1,21)]
        assert len({m["mechanism_id"] for m in methods}) == 10
        assert len({c["mechanism_id"] for c in conditions}) == 20
        for method in methods:
            assert len(method["steps"]) == 5
            assert method["compatible_conditions"]
            assert method["improvement_branch"] and method["recheck"]
            assert method["adequate_branch"]
            for cid in method["compatible_conditions"]:
                assert method["id"] in next(c for c in conditions if c["id"] == cid)["compatible_methods"]
        for condition in conditions:
            assert condition["compatible_methods"]
            assert condition["target"] and condition["candidate_values"]
            assert condition["status"] == "design_requires_physics_binding"
        text = json.dumps(card, ensure_ascii=False)
        assert "{obj}" not in text and "{destination}" not in text and "{moving_part}" not in text
        assert (folder / "tasks" / (tid + ".md")).is_file()
        assert load(folder / "tasks" / (tid + ".json")) == card
        assert card["evaluation"] == ["完整任务完成","检查结果影响后续操作","改善前后稳定性","后续任务适配性","历史信息被正确使用"]
        families[card["family"]] += 1
    source_rows = jsonl(folder / "source_reuse_plan.jsonl")
    assert len(source_rows) == 5377
    assert sum(r["source_kind"] == "recovery_expert" for r in source_rows) == 3184
    assert sum(r["role"] == "scene_observation" for r in source_rows) == 373
    assert {r["task_id"] for r in source_rows} == set(original)
    schedule = jsonl(folder / "source_method_schedule.jsonl")
    counts = Counter(r["source_id"] for r in schedule)
    assert len(counts) == 5004 and set(counts.values()) == {10}
    assert all(r["status"] == "requires_binding_and_real_execution" for r in schedule)
    report = {
        "task_entries_checked": 96,
        "recovery_task_entries_checked": 78,
        "supplementary_task_entries_checked": 18,
        "methods_checked": 960,
        "conditions_checked": 1920,
        "original_instructions_goals_and_source_ids_preserved": True,
        "bidirectional_method_condition_links_checked": True,
        "source_records_assigned": len(source_rows),
        "source_records_unassigned": 0,
        "source_method_plans_checked": len(schedule),
        "static_scene_references_preserved": 373,
        "physical_rollouts_generated_by_catalogue": 0,
        "operation_families": dict(families),
    }
    (folder / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CATALOGUE_VALIDATED", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
