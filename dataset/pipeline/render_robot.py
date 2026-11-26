import argparse
import json
import gc
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def restore(model, data, state):
    expected = 1 + model.nq + model.nv + model.na
    assert len(state) == expected, (len(state), expected)
    data.time = float(state[0])
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
    if model.na:
        data.act[:] = state[-model.na:]
    mujoco.mj_forward(model, data)


def camera(lookat, distance, angle, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = angle
    cam.elevation = elevation
    return cam


def labelled_image(array, text):
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 26], fill=(245, 246, 248))
    draw.text((10, 6), text, fill=(18, 24, 30))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--orbit", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.trajectory.suffix == ".npz":
        with np.load(args.trajectory) as source:
            states = source["before_states"].copy()
            phases = [str(x) for x in source["phase"]]
    else:
        with h5py.File(args.trajectory) as source:
            states = source["observations/state"][:]
            phases = [x.decode() if isinstance(x, bytes) else str(x) for x in source["phase"][:]]
            phases.append("final")
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.vis.global_.offwidth = max(640, int(model.vis.global_.offwidth))
    model.vis.global_.offheight = max(480, int(model.vis.global_.offheight))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.geomgroup[0] = 0
    options.geomgroup[1] = 1
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    wide = camera([-0.15, 0.08, 1.0], 1.55, 145, -25)
    detail = camera([0.02, 0.13, 0.95], 0.70, 140, -40)
    video_path = args.output / "robot_motion.mp4"
    writer = imageio.get_writer(
        video_path, format="FFMPEG", fps=20 / args.stride,
        codec="libx264", quality=7, macro_block_size=16,
        ffmpeg_log_level="error",
    )
    selected = sorted({
        0, len(states)-1,
        next((i for i,phase in enumerate(phases) if phase=="probe"), len(states)//3),
        next((i for i,phase in enumerate(phases) if phase=="regrasp"), 2*len(states)//3),
    })
    for index in range(0, len(states), args.stride):
        restore(model, data, states[index])
        renderer.update_scene(data, camera=wide, scene_option=options)
        left = labelled_image(renderer.render(), f"Full robot | SIMULATION | t={index/20:.2f}s")
        renderer.update_scene(data, camera=detail, scene_option=options)
        right = labelled_image(renderer.render(), f"Contact detail | {phases[index]}")
        combined = np.concatenate((left, right), axis=1)
        writer.append_data(combined)
        if index % 80 == 0:
            print("RENDERED", index, "/", len(states), flush=True)
        for number, chosen in enumerate(selected):
            if index <= chosen < index+args.stride:
                Image.fromarray(combined).save(args.output / f"keyframe_{number}.jpg", quality=92)
    writer.close()
    if args.orbit:
        for number,index in enumerate(selected):
            restore(model, data, states[index])
            for angle_index in range(12):
                cam = camera([-0.15,0.08,1.0], 1.55, angle_index * 30, -25)
                renderer.update_scene(data, camera=cam, scene_option=options)
                image = labelled_image(renderer.render(), f"Actual geometry | t={index/20:.2f}s | angle={angle_index*30}")
                Image.fromarray(image).save(args.output / f"orbit_{number}_{angle_index}.jpg", quality=86)
    del renderer
    gc.collect()
    index_html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>机械臂真实仿真回放</title><style>body{max-width:1250px;margin:30px auto;padding:20px;font:17px/1.7 system-ui}
video,img{max-width:100%;border-radius:8px} label{display:block;margin:15px 0}</style>
<h1>机械臂与物体：真实 MuJoCo 三维回放</h1>
<p>视频按保存的物理状态重新渲染，包含完整机械臂和接触细节。</p>
<video src="robot_motion.mp4" controls preload="metadata"></video>
<h2>关键时刻</h2>""" + "".join(f'<img src="keyframe_{k}.jpg">' for k in range(len(selected)))
    if args.orbit:
        index_html += """<h2>多角度检查</h2><p>选择实际记录的时刻，拖动角度查看同一物理状态的机械臂和物体。</p>
<label>时刻 <input id="time" type="range" min="0" max="3" value="1"></label>
<label>角度 <input id="angle" type="range" min="0" max="11" value="5"></label>
<img id="scene" src="orbit_1_5.jpg"><script>
const t=document.getElementById('time'),a=document.getElementById('angle'),s=document.getElementById('scene');
function update(){s.src='orbit_'+t.value+'_'+a.value+'.jpg'} t.oninput=a.oninput=update;
</script>"""
    (args.output / "index.html").write_text(index_html + "</html>", encoding="utf-8")
    summary = {"frames": len(states), "video":str(video_path),"full_robot":True,
               "native_mujoco_geometry":True,"camera_views":2,"selected_frames":selected,
               "orbit_views":12 if args.orbit else 0}
    (args.output / "render_summary.json").write_text(json.dumps(summary,indent=2))
    print("ROBOT_RENDER_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
