import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-v4.1-flash"


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(item, minimum, maximum):
    return {"type": "array", "items": item, "minItems": minimum, "maxItems": maximum}


TEXT = {"type": "string"}
SETUP = {"type": "string", "enum": ["parameter_change", "layout_change", "new_fixture", "external_force"]}
CONDITION = obj({
    "id": TEXT, "name": TEXT, "target": TEXT, "physical_realization": TEXT,
    "candidate_values": TEXT, "visible_effect": TEXT, "setup": SETUP,
})
METHOD = obj({
    "id": TEXT, "title": TEXT, "phase": TEXT, "question": TEXT,
    "steps": array(TEXT, 4, 6), "observable_evidence": TEXT,
    "adequate_branch": TEXT, "improvement_branch": TEXT, "recheck": TEXT,
    "condition_ids": array(TEXT, 2, 6), "expert_reuse": TEXT,
})
TASK = obj({
    "task_id": TEXT, "title_zh": TEXT, "object_and_goal": TEXT,
    "source_fragments": TEXT, "methods": array(METHOD, 10, 10),
    "conditions": array(CONDITION, 16, 16),
})

INSTRUCTION = """你是机器人数据集设计作者。完成用户要求：对提供的每一个真实任务，逐项设计10种具体、自洽、有区别的自测与主动改善方法，以及16种任务适用的恶劣条件。输出完整JSON，不调用任何工具、不读目录、不创建计划文件、不委派。输入已经包含全部任务来源信息。
研究模型为VLA-Adapter-Pro，双RGB、末端位姿、夹爪宽度、实际动作和8个循环记忆Token。只有生成器能读取物体真值、接触力、质量和摩擦；策略必须用实际视觉和本体反馈。无需触觉。
【本轮明确目标】
以全部已有恢复专家轨迹为基础，充分复用接近、接触、重抓、重放、装置操作和任务完成片段。所有新增设计都包含自测与根据结果改善。允许拿起来以后发现抓得歪，完整放回支撑面、松开、换抓点、重新拿取、复测，也保留检查后直接继续。评价关注完整任务成功、使用反馈、操作质量与历史利用。不要提出控制步数、动作数量、耗时、最短路径、效率优化、代价惩罚或无谓检查率，不因此剔除轨迹。
每个任务10种方法必须与具体物体、目标位置和后续动作紧密结合，至少覆盖抓持/接触建立、姿态或任务适配、支撑或作用效果、复测或历史更新等适用的不同问题。不得用x/y或大小变化把一个摇晃改名充成十种。书要放前/后/左/右隔间、架上/架下等关系严格保留；白杯/红杯、前碗/中碗、先开抽屉再放物/先放物再关抽屉严格保留。没有运输物体的关炉灶、关微波炉任务不能添加搬碗。单轴理想抽屉的“卡歪”需新增真实导轨间隙或挡块并标明new_fixture。每种方法的终点必须是原任务目标。
【物理条件】
16个条件必须作用于具体对象或接触面，包含充分的任务特色。可用低/高滑动摩擦、扭转摩擦、质量/惯量、偏心载荷、浅接触、抓点位置、支撑面、入口间隙、关节frictionloss/damping、真实弹簧回弹、阻挡物、易滚动/旋转、接触遮挡、遮挡后换视角、发生真实碰撞后复测。
用户还希望有大胆的条件，例如“书很滑、书会飞、杯子容易转、抽屉很重”。书会飞应由轻质薄书、受控气流或预先设定的真实外力/弹性机构产生；不许瞬移或直接覆盖坐标。外力时间表在回合开始前固定，以external_force标注。没有新装置支持的液体、书页形变、磁铁、真空、锁扣、热量不能声称原场景已经有；需要时以new_fixture明确建模，主要方法应能复用现有刚体。炉灶只检查实际开关/旋钮可见状态，不写真实发热/煮熟。
候选数值是待仿真校准的设计，不声称实测；单位具体：摩擦/质量/阻尼名义倍数、米、度、牛顿。修改质量同时更新惯量。木柜顶层原始damping=50，使用倍率。滑动摩擦要改指定接触对并保持正确碰撞过滤，不能只改一侧后假定接触生效。世界参数在回合中固定，改善通过机器人动作完成。
【每张任务卡】
methods为M01..M10，conditions为A01..A16；id在各任务内部编号。每个method配2..6个适用condition_ids，每个condition至少被两个不同method使用，形成真实合理的多对多关系。不要随便把无关条件硬连上。每个method的steps为4..6个具体动作句，完整展现检查、发现问题后改善、再次验证与原任务完成；observable_evidence说明相机/本体能看到什么。adequate_branch写当前已足够时继续；improvement_branch写仍可操作但需要改善时如何处理；recheck写怎样验证改善。expert_reuse说明当前任务已有恢复专家中可复用哪段操作。source_fragments在任务层说明怎样利用该任务所有专家，而非只挑一条。
中文通俗、句子直接，每种方法约120..190中文字符，不造黑话；英文标识符保留。不要写“不是…而是…”或任何类似句式。一次完整交付，不提供阶段式方案。你的输出是未执行的设计，数据采集数量由真实运行另行确定。
"""


def run_batch(batch_id, tasks, seconds):
    folder = ROOT / "author_runs" / f"batch_{batch_id:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "answer.json"
    status_file = folder / "run.json"
    if output.is_file() and status_file.is_file():
        recorded = json.loads(status_file.read_text())
        if recorded.get("exit_code") == 0 and not recorded.get("timed_out"):
            print("EXISTING", batch_id, flush=True)
            return recorded
    schema = obj({"tasks": array(TASK, len(tasks), len(tasks))})
    (folder / "schema.json").write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
    evidence = [
        {k: row[k] for k in ("task_id", "source_task", "source_task_indices", "expert_count", "frames")}
        for row in tasks
    ]
    prompt = INSTRUCTION + "\n真实任务如下：\n" + json.dumps(evidence, ensure_ascii=False, indent=2)
    (folder / "TASK.md").write_text(prompt, encoding="utf-8")
    command = [
        "codex", "--no-daemon", "exec", "--skip-git-repo-check", "--ephemeral",
        "-s", "read-only", "-C", str(folder), "-m", MODEL,
        "-c", "features.multi_agent=false",
        "-c", 'model_reasoning_effort="medium"',
        "--json", "--color", "never", "--output-schema", str(folder / "schema.json"),
        "-o", str(output), "-",
    ]
    began = time.monotonic()
    print("START_AUTHOR", batch_id, [row["task_id"] for row in tasks], flush=True)
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
        "batch": batch_id, "tasks": [row["task_id"] for row in tasks],
        "model": MODEL, "no_daemon": True, "effort": "medium",
        "elapsed_seconds": round(time.monotonic() - began, 2),
        "exit_code": process.returncode, "timed_out": timed_out,
        "answer_present": output.is_file(),
    }
    status_file.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print("END_AUTHOR", json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tasks-per-batch", type=int, default=6)
    parser.add_argument("--seconds", type=int, default=720)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--end-batch", type=int)
    args = parser.parse_args()
    assert 1 <= args.workers <= 6
    tasks = json.loads((ROOT / "sources/task_registry.json").read_text())
    assert len(tasks) == 78
    batches = [tasks[i:i + args.tasks_per_batch] for i in range(0, len(tasks), args.tasks_per_batch)]
    selected = list(enumerate(batches))[args.start_batch:args.end_batch]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_batch, bid, batch, args.seconds) for bid, batch in selected]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    (ROOT / "author_results.json").write_text(
        json.dumps(sorted(results, key=lambda item: item["batch"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if any(row["exit_code"] != 0 or row["timed_out"] or not row["answer_present"] for row in results):
        raise RuntimeError("部分作者任务尚未成功交付，请查看author_results.json")
    print("ALL_AUTHOR_BATCHES_FINISHED", len(results), flush=True)


if __name__ == "__main__":
    main()
