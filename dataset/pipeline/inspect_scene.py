import os
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

from vla_branch_lab.assets import configure_libero
from vla_branch_lab.config import load_config

configure_libero(load_config(ROOT / "configs/base.yaml"))
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import mujoco
from PIL import Image

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
env = OffScreenRenderEnv(
    bddl_file_name=suite.get_task_bddl_file_path(0),
    camera_heights=128, camera_widths=128,
    camera_names=["agentview", "robot0_eye_in_hand"],
    control_freq=20, horizon=2000, ignore_done=True, hard_reset=False,
    render_gpu_device_id=3,
)
env.reset()
path = ROOT / "cache/datasets/libero/libero_spatial" / (suite.get_task(0).name + "_demo.hdf5")
with h5py.File(path) as f:
    g = f["data/demo_0"]
    states = g["states"][:]
    actions = g["actions"][:]
    pos = g["obs/ee_pos"][:]
    ori = g["obs/ee_ori"][:]
obs = env.set_init_state(states[0])
raw = env.sim.model._model
print("MUJOCO", mujoco.__version__, flush=True)
print("BODY_NAMES", env.env.obj_body_id, flush=True)
print("GRIPPER", env.robots[0].gripper.important_geoms, flush=True)
c = env.robots[0].controller
print("CONTROLLER", {k: str(getattr(c,k)) for k in ["use_delta","input_max","input_min","output_max","output_min","eef_name"] if hasattr(c,k)}, flush=True)
gi = int(np.flatnonzero(actions[:,6] > 0)[0])
print("GRASP", gi, [{"i": i, "eef": pos[i].tolist(), "bowl": states[i,10:13].tolist(), "ori":ori[i].tolist(), "grip":float(actions[i,6])} for i in range(gi-2,gi+9,2)], flush=True)
xml = env.env.model.get_xml()
Path("scene.xml").write_text(xml)
print("XML", len(xml), "NQ", raw.nq, "raw", type(raw).__name__, flush=True)
for a in actions[:gi+5]:
    obs, _, _, _ = env.step(a)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.9]
cam.distance = 1.75
cam.azimuth = 145
cam.elevation = -27
renderer = mujoco.Renderer(raw, height=512, width=768)
renderer.update_scene(env.sim.data._data, camera=cam)
Image.fromarray(renderer.render()).save("robot_inspect.png")
renderer.close()
env.close()
