import argparse
import html
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    batches=[ROOT/"batch_12",ROOT/"sampled_4"]
    audits=[load(p/"audit.json") for p in batches]
    manifests=[load(p/"manifest.json") for p in batches]
    tuning=load(DATA/"agent_tuning/calibration_results.json")
    verification=load(ROOT/"batch_12/replay_verification.json")
    total={
        "generated_episodes":sum(a["physical_episodes"] for a in audits),
        "complete_task_successes":sum(a["successes"] for a in audits),
        "successful_negative_probe_recoveries":sum(a["recovered_success_episodes"] for a in audits),
        "negative_probe_episodes":sum(a["episodes_with_negative_probe"] for a in audits),
        "physics_steps":sum(a["physical_steps"] for a in audits),
        "conditions_executed":["C01","C03","C04"],
        "methods_executed":["P01","P02","P03","P04","P05","P07"],
        "torch_episode_loader_checked":all(a["dataloader_checked"] for a in audits),
        "physics_replays":verification["replays"],
        "paired_feedback_control":verification["paired_control"],
        "base_vla_training_performed":False,
        "world_changes_during_episode":False,
        "batches":[{"name":p.name,**{k:v for k,v in a.items() if k!="episodes"}} for p,a in zip(batches,audits)],
    }
    (ROOT/"RUN_SUMMARY.json").write_text(json.dumps(total,ensure_ascii=False,indent=2))
    examples=[]
    for folder in batches:
        for row in load(folder/"training_index.json")["episodes"]:
            row={**row,"episode":folder.name+"/"+row["episode"],"trajectory":folder.name+"/"+row["trajectory"]}
            examples.append(row)
    (ROOT/"dataset_index.json").write_text(json.dumps({"episodes":examples,"hidden_truth_excluded":True},ensure_ascii=False,indent=2))
    trials=[]
    for r in tuning["runs"]:
        check=f'{r.get("automatic_checks_passed","—")} / {r.get("automatic_checks_total","—")}'
        trials.append(f'<tr><td>{html.escape(r["run"])}</td><td>{r["wall_seconds"]:.2f}s</td><td>{check}</td><td>{r.get("command_count","—")}</td><td>{"未在限时内交付" if r.get("timed_out") else "已交付"}</td></tr>')
    cases=[]
    for batch,manifest in zip(batches,manifests):
        for r in manifest["records"]:
            first=next(e for e in r["events"] if e["method"].startswith("P"))
            n=sum(e["result"]=="negative" for e in r["events"])
            label=batch.name+"/"+r["plan"]["plan_id"]
            cases.append(f'<tr><td>{label}</td><td>{", ".join(r["plan"]["condition_ids"])}</td><td>{first["method"]}</td><td>{n}</td><td>{r["control_steps"]}</td><td>{"成功" if r["success"] else "失败，保留记录"}</td></tr>')
    galleries=[
        ("试探通过后完成搬运","batch_12/gallery_pass"),
        ("转腕发现问题、重抓、复测后完成","batch_12/gallery_recovery"),
        ("随机抽样任务：试探后调整并完成","sampled_4/gallery_selftest_001"),
    ]
    cards=[]
    for title,rel in galleries:
        assert (ROOT/rel/"index.html").is_file() and (ROOT/rel/"robot_motion.mp4").is_file()
        cards.append(f'<section><h2>{title}</h2><video src="{rel}/robot_motion.mp4" controls preload="metadata"></video><p><a href="{rel}/index.html">查看真实物理状态的12角度回放</a> · <a href="{rel}/robot_motion.mp4" download>保存视频</a></p><img src="{rel}/keyframe_1.jpg"></section>')
    page=f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SelfTest 数据生成与机械臂三维检查</title><style>
body{{font:16px/1.75 system-ui,sans-serif;color:#20272e;background:#f6f7f9;max-width:1260px;margin:24px auto;padding:0 24px 70px}}
h1{{font-size:28px}}h2{{font-size:21px}}section,.summary{{background:#fff;border:1px solid #dfe4ea;border-radius:12px;padding:20px;margin:20px 0}}
video,img{{max-width:100%;border-radius:8px}}video{{width:100%;background:#111}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e2e6ea}}a{{color:#236ba3}}
.scroll{{overflow:auto}}.muted{{color:#586575}}
</style></head><body>
<h1>SelfTest：从任务条件到可训练轨迹</h1>
<p>完整机械臂、物体、接触区域均使用保存的MuJoCo物理状态重新渲染。左侧为机械臂全景，右侧为接触近景。</p>
<div class="summary"><strong>{total["generated_episodes"]}条实际回合 · {total["complete_task_successes"]}条完整成功 · {total["successful_negative_probe_recoveries"]}条发现问题后调整成功</strong>
<p>数据由原始成功专家轨迹与经典反馈控制器生成。回合中的摩擦、质量和惯量保持不变。失败回合完整保留，成功示范单独建立训练索引。</p></div>
{''.join(cards)}
<section><h2>DeepSeek实际任务调试</h2><p>三个相同任务：抓持、抽屉、入口适配。通过直接提供必要证据、限定动作枚举、区分物理配置和回合中操作，完成任务卡生成。</p>
<table><thead><tr><th>任务包</th><th>用时</th><th>自动检查</th><th>终端调用</th><th>结果</th></tr></thead><tbody>{''.join(trials)}</tbody></table>
<p class="muted">24项检查覆盖3张任务卡。物理有效性由随后真实仿真检查；这些数字不代表通用任务可靠率。</p></section>
<section><h2>实际重放与反馈对照</h2>
<p>两条保存的动作序列从原始初态重新执行，完整物理状态最大绝对差均为0，最终任务结果一致。</p>
<p>同一困难初态、同一试探、同一物理条件：使用试探结果调整后成功；忽略反馈继续原路径失败。试探结束前的动作与物理状态一致。这是一个具体情境的机制检查。</p></section>
<section><h2>两批真实结果</h2><div class="scroll"><table><thead><tr><th>回合</th><th>条件</th><th>首测</th><th>未通过次数</th><th>真实步数</th><th>任务结果</th></tr></thead><tbody>{''.join(cases)}</tbody></table></div></section>
<section><h2>训练数据</h2><p>策略输入只包含双RGB、8维本体、过去真实动作、任务文本。未来动作按8步组织，阶段边界使用有效位标记，保留完整时序供循环记忆训练。已通过PyTorch真实数据读取。</p>
<p><a href="dataset_index.json">成功轨迹索引</a> · <a href="RUN_SUMMARY.json">执行摘要</a> · <a href="README.md">运行说明</a></p>
<p>目前物理执行器覆盖低抓持摩擦、质量与惯量变化、抓取偏移三类条件，以及六种检查动作。其他条件仍保留在候选池中，尚未作为已验证轨迹交付。未启动VLA训练。</p></section></body></html>"""
    (ROOT/"report.html").write_text(page,encoding="utf-8")
    with zipfile.ZipFile(ROOT/"SelfTest_Robot_Review.zip","w",zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT/"report.html","report.html")
        z.write(ROOT/"RUN_SUMMARY.json","RUN_SUMMARY.json")
        for _,rel in galleries:
            for p in (ROOT/rel).glob("*"):
                if p.is_file():z.write(p,str(p.relative_to(ROOT)))
        for name in ("dataset_index.json","README.md"):
            if (ROOT/name).exists():z.write(ROOT/name,name)
    print("DELIVERY_READY",json.dumps(total,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
