import argparse
import concurrent.futures as futures
import json
import multiprocessing
import os
import shutil
import time
import zipfile
from collections import Counter
from pathlib import Path

from generate_500 import write_json


def render_job(root, episode):
    from gallery_500 import render_episode
    return render_episode(root, episode)


def create_package(root, folder, counter, kind, items):
    name = f"{counter:06d}_{kind}.zip"
    path = folder / name
    temporary = path.with_suffix(".writing")
    episode_ids = []
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in items:
            episode_id = item["episode"]
            episode_ids.append(episode_id)
            if kind == "data":
                source = root / "episodes" / episode_id / "record"
                for filename in ("trajectory.hdf5", "model.xml", "metadata.json"):
                    archive.write(source / filename, f"dataset/{episode_id}/{filename}")
                source_outcome = root / "episodes" / episode_id / "outcome.json"
                archive.write(source_outcome, f"dataset/{episode_id}/outcome.json")
            else:
                source = root / "preview" / "episodes" / episode_id
                for file in sorted(source.iterdir()):
                    if file.is_file():
                        archive.write(file, f"preview/episodes/{episode_id}/{file.name}")
    temporary.replace(path)
    return {"name": name, "bytes": path.stat().st_size, "kind": kind, "episodes": episode_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--producer-pid", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    while True:
        if (root / "manifest.json").exists():
            current = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if args.producer_pid is None or current["progress"].get("producer_pid") == args.producer_pid:
                break
        time.sleep(2)
    packages_dir = root / "delivery/packages"
    snapshots_dir = root / "delivery/snapshots"
    packages_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    (root / "publisher.pid").write_text(str(os.getpid()))
    manifest_path = root / "delivery/manifest.json"
    packages = []
    if manifest_path.exists():
        packages = json.loads(manifest_path.read_text(encoding="utf-8"))["packages"]
    packed_data = {e for p in packages if p["kind"] == "data" for e in p["episodes"]}
    packed_preview = {e for p in packages if p["kind"] == "preview" for e in p["episodes"]}
    renders = {}
    for path in (root / "preview/episodes").glob("*/rendered.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        renders[result["episode"]] = result
    rendering_errors = {}
    count = len(packages)
    version = json.loads(manifest_path.read_text(encoding="utf-8"))["version"] if manifest_path.exists() else 0
    previous_signature = None
    context = multiprocessing.get_context("spawn")
    with futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        pending = {}
        while True:
            source = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            progress = source["progress"]
            rows = source["records"]
            selected = set(source["selected"])
            indexed = {row["episode"]: row for row in rows}
            desired = set(selected)
            # 额外保留每种布局至多两条失败回放，方便用户审查困难情形。
            failed = Counter()
            for row in rows:
                task = row["plan"]["task_id"]
                if row["simulation_completed"] and not row["eligible"] and failed[task] < 2:
                    desired.add(row["episode"])
                    failed[task] += 1
            scheduled = set(pending.values())
            unrendered = sorted(desired - set(renders) - scheduled - set(rendering_errors))
            # 优先展示负反馈后的成功，以及此前尚未预览的布局。
            shown_tasks = {r["task_id"] for r in renders.values()}
            unrendered.sort(key=lambda e: (
                indexed[e]["plan"]["task_id"] in shown_tasks,
                not (indexed[e].get("audit", {}).get("negative_tests", 0) and indexed[e]["eligible"]),
                e,
            ))
            while unrendered and len(pending) < args.workers:
                episode = unrendered.pop(0)
                future = pool.submit(render_job, str(root), episode)
                pending[future] = episode

            if pending:
                done, _ = futures.wait(pending, timeout=1, return_when=futures.FIRST_COMPLETED)
                for future in done:
                    episode = pending.pop(future)
                    try:
                        renders[episode] = future.result()
                    except Exception as error:
                        rendering_errors[episode] = f"{type(error).__name__}: {error}"
                        print("RENDER_ERROR", episode, rendering_errors[episode], flush=True)
            new_data = sorted(selected - packed_data)
            new_preview = sorted(set(renders) - packed_preview)
            if new_data:
                for start in range(0, len(new_data), 10):
                    counter = count + 1
                    ids = new_data[start:start+10]
                    package = create_package(root, packages_dir, counter, "data", [indexed[e] for e in ids])
                    packages.append(package)
                    packed_data.update(ids)
                    count = counter
            if new_preview:
                counter = count + 1
                package = create_package(root, packages_dir, counter, "preview", [renders[e] for e in new_preview])
                packages.append(package)
                packed_preview.update(new_preview)
                count = counter
            completed = progress["generation_finished"] and not pending and desired <= (set(renders) | set(rendering_errors))
            signature = (progress["attempts_completed"], progress["accepted_successes"], len(renders), len(packages), completed)
            if signature != previous_signature:
                from gallery_500 import dashboard
                version += 1
                snapshot = f"snapshot_{version:06d}.html"
                cards = [{
                    key: result[key] for key in (
                        "episode", "success", "has_negative", "task_id",
                        "first_method", "conditions", "duration_s", "source_demo", "task_label",
                    )
                } for episode,result in renders.items() if episode in desired]
                dashboard(progress, cards, root / "preview/index.html")
                shutil.copy2(root / "preview/index.html", snapshots_dir / snapshot)
                entry_index = json.loads((root / "training_index.json").read_text(encoding="utf-8"))
                for item in entry_index["episodes"]:
                    item["trajectory"] = f"dataset/{item['episode']}/trajectory.hdf5"
                index_name = f"dataset_{version:06d}.json"
                write_json(snapshots_dir / index_name, entry_index)
                write_json(manifest_path, {
                    "run_name": root.name,
                    "version": version,
                    "snapshot": snapshot,
                    "dataset_index": index_name,
                    "progress": progress,
                    "preview_count": len(renders),
                    "packages": packages,
                    "all_finished": completed,
                    "rendering_errors": rendering_errors,
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                print("PUBLISHED", progress["accepted_successes"], "previews", len(renders), "packages", len(packages), flush=True)
                previous_signature = signature
            if completed:
                break
            time.sleep(2)
    print("PUBLICATION_FINISHED", len(packed_data), len(packed_preview), flush=True)


if __name__ == "__main__":
    main()
