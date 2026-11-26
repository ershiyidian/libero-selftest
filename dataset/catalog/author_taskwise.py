import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import jsonschema
from author_catalog import INSTRUCTION, TASK, obj, array, MODEL

ROOT = Path(__file__).resolve().parent

EXTRA = """
本次只写一个任务，必须完整输出10种方法与16个条件，不省略任何字段。
每种方法要有phase和question。steps的每项使用一句具体动作，少用背景说明。
记忆作用用“记住此前试过的抓点/接触/观察结果”表述，不把手工改写神经网络Token当作机器人动作。
source_fragments和expert_reuse属于建议提取的专家片段，不声称已经从源轨迹测得某种现象。
使用提供的BDDL对象和目标谓词核实原任务，不把目标前后的邻近物体弄反。
例如butter at the back表示拿原先靠后的黄油，不能改成拿任意黄油再放到抽屉后面。
比较改善前后的稳定与任务适配，但不以执行次数或时间筛选。保留充分试探和重复调整。
"""


def check_answer(answer, task_id):
    jsonschema.Draft202012Validator(obj({"tasks": array(TASK, 1, 1)})).validate(answer)
    task = answer["tasks"][0]
    assert task["task_id"] == task_id
    assert {m["id"] for m in task["methods"]} == {f"M{x:02d}" for x in range(1, 11)}
    assert {c["id"] for c in task["conditions"]} == {f"A{x:02d}" for x in range(1, 17)}
    conditions = {c["id"] for c in task["conditions"]}
    seen = set()
    for method in task["methods"]:
        assert set(method["condition_ids"]) <= conditions
        seen.update(method["condition_ids"])
    assert seen == conditions, f"没有方法关联的条件：{conditions - seen}"
    assert len({m["title"] for m in task["methods"]}) == 10
    return task


def run_task(task, binding, seconds):
    tid = task["task_id"]
    folder = ROOT / "task_authors" / tid
    folder.mkdir(parents=True, exist_ok=True)
    answer_path = folder / "answer.json"
    accepted_path = ROOT / "task_cards" / f"{tid}.json"
    if accepted_path.is_file():
        return {"task_id": tid, "status": "already_complete"}
    evidence = {key: task[key] for key in ("task_id", "source_task", "source_task_indices", "expert_count", "frames")}
    evidence["source_goal_bindings"] = [
        {key: match[key] for key in ("language", "objects", "objects_of_interest", "goal_predicates")}
        for match in binding.get("exact_language_matches", [])[:2]
    ]
    prompt = INSTRUCTION + EXTRA + "\n真实任务和源任务目标：\n" + json.dumps(evidence, ensure_ascii=False)
    (folder / "TASK.md").write_text(prompt, encoding="utf-8")
    schema = obj({"tasks": array(TASK, 1, 1)})
    (folder / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    command = [
        "codex", "--no-daemon", "exec", "--skip-git-repo-check", "--ephemeral",
        "-s", "read-only", "-C", str(folder), "-m", MODEL,
        "-c", "features.multi_agent=false", "-c", 'model_reasoning_effort="medium"',
        "--json", "--color", "never", "--output-schema", str(folder / "schema.json"),
        "-o", str(answer_path), "-",
    ]
    began = time.monotonic()
    print("START_TASK", tid, flush=True)
    with (folder / "events.jsonl").open("w") as events, (folder / "stderr.log").open("w") as errors:
        process = subprocess.Popen(
            command, cwd=folder, stdin=subprocess.PIPE, stdout=events, stderr=errors,
            text=True, start_new_session=True,
        )
        timed_out = False
        try:
            process.communicate(input=prompt, timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=15)
    record = {
        "task_id": tid, "elapsed_seconds": round(time.monotonic() - began, 2),
        "exit_code": process.returncode, "timed_out": timed_out,
        "model": MODEL, "no_daemon": True, "status": "no_answer",
    }
    if process.returncode == 0 and not timed_out and answer_path.is_file():
        try:
            answer = json.loads(answer_path.read_text())
            card = check_answer(answer, tid)
            accepted_path.parent.mkdir(parents=True, exist_ok=True)
            accepted_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
            record["status"] = "structurally_valid"
        except (json.JSONDecodeError, jsonschema.ValidationError, AssertionError) as error:
            record["status"] = "needs_revision"
            record["error"] = str(error)
    (folder / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("END_TASK", json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seconds", type=int, default=360)
    parser.add_argument("--task-ids", nargs="*")
    args = parser.parse_args()
    assert 1 <= args.workers <= 6
    tasks = json.loads((ROOT / "sources/task_registry.json").read_text())
    bindings = {row["task_id"]: row for row in json.loads((ROOT / "sources/recovery_to_original_tasks.json").read_text())}
    if args.task_ids:
        tasks = [row for row in tasks if row["task_id"] in args.task_ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, task, bindings[task["task_id"]], args.seconds) for task in tasks]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    (ROOT / "task_author_results.json").write_text(
        json.dumps(sorted(results, key=lambda row: row["task_id"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("COMPLETE_TASKS", sum(row["status"] in ("already_complete", "structurally_valid") for row in results),
          "OF", len(results), flush=True)


if __name__ == "__main__":
    main()
