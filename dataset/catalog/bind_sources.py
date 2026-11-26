import argparse
import importlib.util
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/guest/vla_branch_lab/VLA-SelfTest")
DATA = ROOT / "data_generation"
OUT = DATA / "all_tasks/sources"
LIBERO = ROOT.parent / "external/LIBERO/libero/libero"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    spec = importlib.util.spec_from_file_location("official_bddl_utils", LIBERO / "envs/bddl_utils.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    tasks = json.loads((OUT / "task_registry.json").read_text())
    language_map = defaultdict(list)
    documents = []
    for path in sorted((LIBERO / "bddl_files").glob("*/*.bddl")):
        parsed = parser.robosuite_parse_problem(str(path))
        language = parsed["language_instruction"]
        if isinstance(language, list):
            language = " ".join(language)
        item = {
            "source": str(path), "language": language, "suite": path.parent.name,
            "objects": parsed["objects"], "fixtures": parsed["fixtures"],
            "objects_of_interest": parsed["obj_of_interest"],
            "goal_predicates": parsed["goal_state"],
            "initial_predicates": parsed["initial_state"],
        }
        documents.append(item)
        language_map[language].append(item)
    write_json(OUT / "original_task_bindings.json", documents)
    task_bindings = []
    for task in tasks:
        matches = language_map.get(task["source_task"], [])
        task_bindings.append({
            "task_id": task["task_id"],
            "source_task": task["source_task"],
            "exact_language_matches": matches,
            "requires_alias_review": not bool(matches),
        })
    write_json(OUT / "recovery_to_original_tasks.json", task_bindings)
    print("BDDL_TASK_BINDINGS", len(documents), "exact_recovery_matches",
          sum(bool(row["exact_language_matches"]) for row in task_bindings), flush=True)

    archive_reports = []
    for path in sorted((DATA / "sources/recovery_history").glob("*.tar.gz")):
        levels = Counter()
        names = []
        count = 0
        size = 0
        with tarfile.open(path, "r:*") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                count += 1
                size += member.size
                pieces = Path(member.name).parts
                key = "/".join(pieces[:min(4, len(pieces) - 1)])
                levels[key] += 1
                if len(names) < 12:
                    names.append(member.name)
        report = {
            "archive": str(path), "file_count": count, "unpacked_bytes": size,
            "sample_members": names, "directory_counts": dict(levels),
            "reader": "tarfile r:*", "all_headers_read": True,
        }
        archive_reports.append(report)
        print("HISTORY_ARCHIVE", path.name, count, size, flush=True)
    write_json(OUT / "history_archive_index.json", archive_reports)


if __name__ == "__main__":
    main()
