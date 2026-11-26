import argparse
import base64
import gzip
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path("/data/guest/vla_branch_lab/VLA-SelfTest")
DATA = ROOT / "data_generation"
sys.path.insert(0, str(DATA / "quality/deps"))
import pyarrow.parquet as pq


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def collect(output):
    output.mkdir(parents=True, exist_ok=True)
    expert = DATA / "sources/full_expert"
    labels = jsonl(expert / "meta/tasks.jsonl")
    episodes = jsonl(expert / "meta/episodes.jsonl")
    index_to_name = {row["task_index"]: row["task"] for row in labels}
    assert len(index_to_name) == len(labels)
    group_map = {}
    for row in labels:
        name = row["task"]
        if name not in group_map:
            group_map[name] = {
                "task_id": f"T{len(group_map) + 1:03d}", "source_task": name,
                "source_task_indices": [], "episode_ids": [], "frames": 0,
            }
        group_map[name]["source_task_indices"].append(row["task_index"])
    registry = []
    faults = []
    all_seen = set()
    for episode in episodes:
        eid = int(episode["episode_index"])
        assert eid not in all_seen
        all_seen.add(eid)
        chunk = f"chunk-{eid // 1000:03d}"
        filename = f"episode_{eid:06d}"
        parquet = expert / "data" / chunk / f"{filename}.parquet"
        videos = {
            camera: expert / "videos" / chunk / camera / f"{filename}.mp4"
            for camera in ("observation.images.image", "observation.images.wrist_image")
        }
        metadata = pq.read_metadata(parquet)
        actual_indices = sorted(set(
            pq.read_table(parquet, columns=["task_index"])["task_index"].to_pylist()
        ))
        names = list(episode["tasks"])
        for name in names:
            assert name in group_map, (eid, name)
            assert name in {index_to_name[idx] for idx in actual_indices}, (eid, actual_indices, name)
            group_map[name]["episode_ids"].append(eid)
            group_map[name]["frames"] += metadata.num_rows
        missing = [str(path) for path in videos.values() if not path.is_file()]
        length_matches = metadata.num_rows == episode["length"]
        if missing or not length_matches:
            faults.append({"episode": eid, "missing_videos": missing, "length_matches": length_matches})
        registry.append({
            "source_kind": "recovery_expert", "episode_id": eid,
            "task_ids": [group_map[name]["task_id"] for name in names],
            "source_task_indices": actual_indices, "source_tasks": names,
            "frames": metadata.num_rows,
            "trajectory": str(parquet.relative_to(ROOT)),
            "videos": {name: str(path.relative_to(ROOT)) for name, path in videos.items()},
            "available": not missing and length_matches,
            "reuse_roles": (
                ["initial_observation", "scene_reference"]
                if metadata.num_rows < 2 else
                ["recovery_suffix", "contact_approach", "support_regrasp_reference",
                 "task_completion_reference", "test_insertion_candidate"]
            ),
            "replay_binding": "match_task_objects_and_saved_state_before_replay",
        })
    with (output / "recovery_sources.jsonl").open("w", encoding="utf-8") as target:
        for row in registry:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    groups = list(group_map.values())
    for row in groups:
        row["expert_count"] = len(row["episode_ids"])
        row["registry"] = "recovery_sources.jsonl"
    assert len(groups) == 78 and len(registry) == 3184
    assert not faults, faults

    original = []
    hdf5_root = ROOT.parent / "cache/datasets/libero"
    for filename in sorted(hdf5_root.glob("*/*_demo.hdf5")):
        with h5py.File(filename) as source:
            for name, demo in source["data"].items():
                original.append({
                    "source_kind": "original_hdf5", "suite": filename.parent.name,
                    "task_file": str(filename), "demo": name,
                    "frames": len(demo["actions"]),
                    "full_states": "states" in demo,
                    "reuse_roles": ["full_prefix", "full_task_demonstration", "simulation_replay", "task_goal_reference"],
                })
    write_json(output / "original_hdf5_sources.json", original)

    rlds = []
    for filename in sorted((ROOT / "official_training/assets/libero").rglob("dataset_info.json")):
        info = json.loads(filename.read_text())
        splits = info.get("splits", [])
        rlds.append({
            "dataset": info.get("name", filename.parent.parent.name),
            "metadata_path": str(filename),
            "episode_count_from_metadata": sum(sum(int(value) for value in x.get("shardLengths", [])) for x in splits),
            "shard_paths": [str(p) for p in sorted(filename.parent.glob("*.tfrecord*"))],
            "reuse_roles": ["ordinary_policy_supervision", "temporal_visual_action_reference"],
            "replay_binding": "do_not_assume_rlds_contains_full_mujoco_state",
        })
    write_json(output / "rlds_sources.json", rlds)

    scene_dir = DATA / "sources/recovery_assets"
    archives = []
    for filename in sorted(scene_dir.glob("*.tar.gz")):
        count = Counter()
        with tarfile.open(filename, "r:gz") as bundle:
            for member in bundle:
                if member.isfile():
                    if member.name.endswith(".bddl"):
                        count["task_descriptions"] += 1
                    elif member.name.endswith("_sim_state.npy"):
                        count["physics_states"] += 1
        archives.append({
            "archive": str(filename.relative_to(ROOT)), "bytes": filename.stat().st_size,
            "members": dict(count), "reuse_roles": ["actual_recoverable_scene", "challenging_state_source"],
        })
    history = []
    for filename in sorted((DATA / "sources/recovery_history").glob("*.gz*")):
        history.append({
            "path": str(filename.relative_to(ROOT)), "bytes": filename.stat().st_size,
            "complete_archive_file": filename.name.endswith(".gz"),
            "reuse_roles": ["pre_failure_history", "locate_earlier_test_opportunity"],
        })
    write_json(output / "task_registry.json", groups)
    summary = {
        "recovery_task_indices": len(labels), "unique_source_task_texts": len(groups),
        "recovery_episodes": len(registry),
        "recovery_frames": sum(row["frames"] for row in registry),
        "recovery_source_file_checks_passed": len(registry) - len(faults),
        "recovery_static_only_examples": sum(row["frames"] < 2 for row in registry),
        "original_hdf5_demonstrations": len(original),
        "original_hdf5_tasks": len({x["task_file"] for x in original}),
        "rlds_datasets": rlds,
        "scene_archives": archives, "history_archives": history,
        "source_episode_count_is_not_generated_selftest_count": True,
    }
    write_json(output / "source_inventory.json", summary)
    compact = [
        {key: row[key] for key in ("task_id", "source_task", "source_task_indices", "expert_count", "frames")}
        for row in groups
    ]
    write_json(output / "task_brief.json", compact)
    print("SOURCE_INVENTORY", json.dumps(summary, ensure_ascii=False), flush=True)
    print("TASK_BRIEF", json.dumps(compact, ensure_ascii=False), flush=True)
    print("TASK_REGISTRY_GZIP_BASE64", base64.b64encode(gzip.compress(
        json.dumps(groups, ensure_ascii=False).encode()
    )).decode(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DATA / "all_tasks/sources")
    collect(parser.parse_args().output)
