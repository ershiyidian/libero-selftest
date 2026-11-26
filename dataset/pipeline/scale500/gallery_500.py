import gc
import html
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from render_robot import restore, camera, labelled_image
from planning_500 import METHOD_NAMES, CONDITION_NAMES


TASK_NAMES = {
    0: "碗位于盘子与烤盅之间", 1: "碗位于烤盅旁",
    2: "碗位于桌面中间", 3: "碗位于饼干盒上",
    4: "从柜子上层抽屉取碗", 5: "碗位于烤盅上",
    6: "碗位于饼干盒旁", 7: "碗位于炉灶上",
    8: "碗位于盘子旁", 9: "碗位于柜子上",
}
STAGES = {
    "expert_prefix": "专家前段", "stabilize": "保持位置",
    "probe": "物理自测", "regrasp": "调整抓取", "transport": "搬运",
    "place": "放置", "release": "松开与撤回", "final": "任务结束",
}
CSS = """
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#182432;background:#f5f7fa}
body{max-width:1320px;margin:0 auto;padding:28px 22px 70px;line-height:1.6}
h1{font-size:28px;margin:8px 0}h2{font-size:20px;margin-top:30px}h3{font-size:16px}
a{color:#175d95;text-decoration:none}p{margin:8px 0}.muted{color:#617080;font-size:14px}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:24px 0}
.stat,.card,.box{background:white;border:1px solid #e0e6ee;border-radius:12px;padding:16px}
.stat strong{display:block;font-size:27px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
select,input,button{padding:9px 12px;border:1px solid #c9d3dd;border-radius:7px;background:white;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin-top:20px}
.card{padding:0;overflow:hidden}.card img{width:100%;aspect-ratio:8/3;object-fit:contain;background:#eef1f4}
.content{padding:12px 15px}.badge{font-size:12px;padding:3px 7px;background:#ecf4f0;border-radius:5px}
.bad{background:#f9edeb}.row{display:flex;justify-content:space-between;gap:12px}
img,video{max-width:100%;height:auto;border-radius:8px}video{background:#141414;width:100%}
table{width:100%;border-collapse:collapse;font-size:14px}td,th{border-bottom:1px solid #e2e7ed;text-align:left;padding:9px}
table th{color:#4b5a69}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}
progress{width:100%;height:14px;accent-color:#267867}
@media(max-width:900px){.stats{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:600px){.grid,.twocol{grid-template-columns:1fr}body{padding:18px 12px}}
"""


def esc(x):
    return html.escape(str(x), quote=True)


def render_episode(run_string: str, episode_id: str, stride: int = 3) -> dict:
    root = Path(run_string)
    record = root / "episodes" / episode_id / "record"
    meta = json.loads((record / "metadata.json").read_text(encoding="utf-8"))
    out = root / "preview" / "episodes" / episode_id
    out.mkdir(parents=True, exist_ok=True)
    if (out / "rendered.json").is_file():
        return json.loads((out / "rendered.json").read_text())
    with h5py.File(record / "trajectory.hdf5") as file:
        states = file["observations/state"][:]
        phases = file["phase"].asstr()[:].tolist() + ["final"]
        object_xyz = file["observations/bowl"][:]
    model = mujoco.MjModel.from_xml_path(str(record / "model.xml"))
    model.vis.global_.offwidth = max(480, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(360, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=360, width=480)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[0] = 0
    options.geomgroup[1] = 1
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    wide = camera([-0.15, 0.04, 1.08], 1.90, 145, -23)
    events = meta["events"]
    tests = [item for item in events if item["method"].startswith("P")]
    negative = next((item for item in tests if item["result"] == "negative"), None)
    selected = sorted(set([
        0, tests[0]["start_step"],
        (negative or tests[0])["end_step"], len(states) - 1,
    ]))
    writer = imageio.get_writer(
        out / "robot_motion.mp4", format="FFMPEG", fps=20 / stride,
        codec="libx264", quality=6, macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-threads", "1", "-movflags", "+faststart"],
    )

    def image_at(index):
        restore(model, data, states[index])
        renderer.update_scene(data, camera=wide, scene_option=options)
        left = labelled_image(renderer.render(), f"Robot | {index/20:.2f} s")
        center = object_xyz[index].copy()
        center[2] = max(0.85, min(1.12, center[2] + 0.035))
        center[:2] = np.clip(center[:2], -0.55, 0.5)
        detail = camera(center, 0.64, 140, -32)
        renderer.update_scene(data, camera=detail, scene_option=options)
        right = labelled_image(renderer.render(), phases[index])
        return np.concatenate((left, right), axis=1)

    for index in range(0, len(states), stride):
        writer.append_data(image_at(index))
    if (len(states) - 1) % stride:
        writer.append_data(image_at(len(states) - 1))
    writer.close()
    for number, index in enumerate(selected):
        Image.fromarray(image_at(index)).save(out / f"key_{number}.jpg", quality=85)
    Image.open(out / "key_1.jpg").resize((640, 240)).save(out / "poster.jpg", quality=78)
    # 保留两个真实时刻的环绕观察：测试结果和最终放置。
    for moment, index in enumerate((selected[-2], selected[-1])):
        restore(model, data, states[index])
        for angle in range(8):
            orbit = camera([-0.15, 0.04, 1.08], 1.9, angle * 45, -23)
            renderer.update_scene(data, camera=orbit, scene_option=options)
            Image.fromarray(renderer.render()).save(out / f"orbit_{moment}_{angle}.jpg", quality=80)
    del renderer
    gc.collect()
    plan = meta["plan"]
    detail_rows = []
    for event in tests:
        outcome = {"pass": "通过", "negative": "发现问题", "inconclusive": "需要补充检查"}[event["result"]]
        detail_rows.append(f"<tr><td>{METHOD_NAMES[event['method']]}</td><td>{event['start_step']/20:.2f}–{event['end_step']/20:.2f}秒</td><td>{outcome}</td><td>{event['relative_shift_m']*1000:.2f}毫米</td></tr>")
    physics = plan["physics"]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{episode_id} · 自测回放</title><style>{CSS}</style>
<a href="../../index.html">← 返回全部样本</a>
<h1>{esc(episode_id)} · {esc(TASK_NAMES[plan['task_id']])}</h1>
<p>原任务{"成功" if meta["success"] else "失败"} · 共{meta["control_steps"]/20:.1f}秒 · 测试{len(tests)}次 · 重新抓取{sum(e["method"]=="regrasp" for e in events)}次</p>
<p class="muted">完整机械臂与接触近景均由此回合实际物理状态重新渲染。图中 Robot 为机械臂，probe 为自测，regrasp 为重新抓取。</p>
<video controls playsinline preload="metadata" poster="poster.jpg" src="robot_motion.mp4"></video>
<h2>测试与反馈</h2><table><tr><th>方法</th><th>时段</th><th>结果</th><th>物体相对夹爪变化</th></tr>{''.join(detail_rows)}</table>
<h2>物理条件</h2><div class="box">滑动摩擦：{physics['friction_multiplier']}倍 · 扭转摩擦：{physics['torsional_multiplier']}倍 · 质量及惯量：{physics['mass_multiplier']}倍 · 初次抓点偏移：{physics['grasp_offset_m']*1000:g}毫米</div>
<h2>同一物理状态的多个观察角度</h2>
<div class="controls"><label>时刻 <select id="moment"><option value="0">测试结果</option><option value="1">最终状态</option></select></label><label>角度 <input id="angle" type="range" min="0" max="7" value="3"></label></div>
<img id="orbit" src="orbit_0_3.jpg" alt="实际机械臂三维场景">
<h2>关键时刻</h2><div class="twocol">{''.join(f'<div><p>{index/20:.2f}秒 · {STAGES.get(phases[index], phases[index])}</p><img loading="lazy" src="key_{n}.jpg"></div>' for n,index in enumerate(selected))}</div>
<h2>源轨迹与数据</h2><p>任务编号 {plan["task_id"]}，专家示范 {plan["demo_id"]}，分组 {esc(plan["split_group"])}，数据划分 {esc(plan["split"])}。</p>
<p class="muted">测试反馈由离线生成器读取物理状态后产生，VLA尚未参与本回合控制。监督读取器仅提供实际图像、本体状态和动作历史。</p>
<script>function update(){{document.getElementById('orbit').src='orbit_'+document.getElementById('moment').value+'_'+document.getElementById('angle').value+'.jpg'}}document.getElementById('moment').onchange=update;document.getElementById('angle').oninput=update;</script></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    result = {
        "episode": episode_id, "frames": len(states), "stride": stride,
        "success": bool(meta["success"]), "has_negative": negative is not None,
        "task_id": plan["task_id"], "first_method": plan["first_method"],
        "conditions": plan["condition_ids"], "duration_s": meta["control_steps"]/20,
        "source_demo": plan["demo_id"], "task_label": TASK_NAMES[plan["task_id"]],
        "metadata": meta,
    }
    (out / "rendered.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def dashboard(progress: dict, cards: list[dict], output: Path) -> None:
    labels = {"running": "正在生成", "complete": "生成完成", "stopped": "已停止", "attempt_limit": "已达到尝试上限"}
    p = progress
    data = json.dumps(cards, ensure_ascii=False).replace("<", "\\u003c")
    options = ''.join(f'<option value="{k}">{v}</option>' for k,v in TASK_NAMES.items())
    method_options = ''.join(f'<option value="{k}">{k} {v}</option>' for k,v in METHOD_NAMES.items())
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SelfTest 500 · 样本抽查</title><style>""" + CSS + """</style>
<h1>SelfTest 500 · 样本抽查</h1>
<p class="muted">双击桌面入口即可查看。此页面每60秒刷新；单个样本的回放页面不会自动刷新。</p>
<div class="row"><strong>""" + labels[p["state"]] + """</strong><span>服务端更新 """ + esc(p["updated_utc"]) + """</span></div>
<progress max="500" value=""" + str(p["accepted_successes"]) + """></progress>
<div class="stats">""" + ''.join(f'<div class="stat">{label}<strong>{value}</strong></div>' for label,value in [
    ("完整成功示范",f'{p["accepted_successes"]} / {p["target_successes"]}'),
    ("已完成尝试",p["attempts_completed"]),("发现问题后成功",p["successful_recoveries"]),
    ("可播放预览",len(cards)),("不同专家历史",p["unique_expert_histories"])]) + """</div>
<div class="box"><strong>抽查内容</strong><p>10种空间布局，4类物理条件，6种测试方法，随机运动幅度、方向及持续时间。保存全部失败尝试；成功示范单独建立索引。生成与渲染使用CPU，不加载训练模型。</p>
<p class="muted">任务仍属于LIBERO-Spatial黑碗抓放。本批扩大布局、历史及检查组合，其他物体和抽屉方法尚未计入此批。</p></div>
<h2>完整机械臂回放</h2><div class="controls">
<select id="task"><option value="">全部任务</option>""" + options + """</select>
<select id="method"><option value="">全部首测</option>""" + method_options + """</select>
<select id="outcome"><option value="">全部结果</option><option value="recovered">发现问题后成功</option><option value="direct">测试通过后成功</option><option value="failed">任务失败</option></select>
<input id="search" placeholder="搜索样本编号"><button id="reset">清除筛选</button>
</div><p id="count" class="muted"></p><div id="grid" class="grid"></div>
<div class="controls" style="margin-top:25px"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<h2>本地文件</h2><p>预览位于 <code>preview/episodes/</code>；完整成功轨迹位于 <code>dataset/</code>。原始训练数据与预览会分批传输到此目录。</p>
<script>const rows=""" + data + r""";let page=0;const size=24;
const ids=['task','method','outcome','search'];function value(id){return document.getElementById(id).value;}
function render(){let filtered=rows.filter(r=>(!value('task')||r.task_id==value('task'))&&(!value('method')||r.first_method==value('method'))&&(!value('search')||r.episode.includes(value('search')))&&(!value('outcome')||(value('outcome')=='recovered'&&r.success&&r.has_negative)||(value('outcome')=='direct'&&r.success&&!r.has_negative)||(value('outcome')=='failed'&&!r.success)));
filtered.sort((a,b)=>b.episode.localeCompare(a.episode));const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.max(0,Math.min(page,pages-1));document.getElementById('count').textContent='匹配 '+filtered.length+' 条，显示 '+(page+1)+' / '+pages+' 页';
document.getElementById('grid').innerHTML=filtered.slice(page*size,(page+1)*size).map(r=>`<a class="card" href="episodes/${r.episode}/index.html"><img loading="lazy" src="episodes/${r.episode}/poster.jpg"><div class="content"><div class="row"><b>${r.episode}</b><span class="badge ${r.success?'':'bad'}">${r.success?(r.has_negative?'调整后成功':'成功'):'失败'}</span></div><p>${r.task_label}</p><p class="muted">${r.first_method} · ${r.duration_s.toFixed(1)}秒 · 源示范 ${r.source_demo}</p></div></a>`).join('');
document.getElementById('page').textContent=(page+1)+' / '+pages;
}
ids.forEach(id=>document.getElementById(id).oninput=()=>{page=0;render()});document.getElementById('reset').onclick=()=>{ids.forEach(id=>document.getElementById(id).value='');page=0;render();};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};render();setTimeout(()=>location.reload(),60000);
</script></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".writing")
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(output)
