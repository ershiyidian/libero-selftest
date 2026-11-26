import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "VLA-SelfTest"
DATA = PROJECT / "data_generation"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

from vla_branch_lab.assets import configure_libero
from vla_branch_lab.config import load_config

configure_libero(load_config(ROOT / "configs/base.yaml"))
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

TARGET = "akita_black_bowl_1"
SUPPORTED_CONDITIONS = ("C01", "C03", "C04")
SUPPORTED_METHODS = ("P01", "P02", "P03", "P04", "P05", "P07")


def vector(text):
    return np.fromstring(text, sep=" ")


def words(values):
    return " ".join(f"{float(x):.12g}" for x in values)


def make_env(task_id, seed, condition):
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_id),
        camera_heights=128, camera_widths=128,
        camera_names=["agentview", "robot0_eye_in_hand"],
        control_freq=20, horizon=2400, ignore_done=True, hard_reset=False,
        render_gpu_device_id=int(os.environ["MUJOCO_EGL_DEVICE_ID"]),
    )
    env.seed(seed)
    env.reset()
    original = env.sim.model
    raw = original._model
    body_id = env.env.obj_body_id[TARGET]
    body_name = original.body_id2name(body_id)
    xml = ET.fromstring(env.env.model.get_xml())
    # LIBERO在reset时直接设置固定装置位置，源XML没有这些运行时变更。
    # 重新编译前保存实际位姿，避免柜体和炉灶回到原点后撞飞桌面物体。
    for element in xml.findall(".//worldbody//body"):
        name = element.get("name")
        if name is not None:
            index = original.body_name2id(name)
            element.set("pos", words(raw.body_pos[index]))
            for attr in ("euler", "axisangle", "xyaxes", "zaxis"):
                element.attrib.pop(attr, None)
            element.set("quat", words(raw.body_quat[index]))
    body = xml.find(f".//body[@name='{body_name}']")
    assert body is not None
    inertia = body.find("inertial")
    if inertia is None:
        inertia = ET.SubElement(body, "inertial")
    mass_scale = float(condition.get("mass_multiplier", 1.0))
    inertia.attrib.clear()
    inertia.attrib.update({
        "pos": words(raw.body_ipos[body_id]),
        "quat": words(raw.body_iquat[body_id]),
        "mass": str(float(raw.body_mass[body_id]) * mass_scale),
        "diaginertia": words(raw.body_inertia[body_id] * mass_scale),
    })
    target_geoms = [
        geom for geom in body.iter("geom")
        if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1"))
    ]
    fingers = sorted({
        name for group in ("left_finger", "right_finger")
        for name in env.robots[0].gripper.important_geoms[group]
    })
    contact = xml.find("contact")
    if contact is None:
        contact = ET.SubElement(xml, "contact")
    ratio = float(condition.get("friction_multiplier", 1.0))
    pair_names = []
    for tg in target_geoms:
        ti = original.geom_name2id(tg.attrib["name"])
        for name in fingers:
            fi = original.geom_name2id(name)
            nominal = np.maximum(raw.geom_friction[ti], raw.geom_friction[fi])
            # 显式接触对只改变目标物体和指尖，不改变桌面等其他接触。
            friction = [nominal[0] * ratio, nominal[0] * ratio, nominal[1], nominal[2], nominal[2]]
            weights = np.array([raw.geom_solmix[ti], raw.geom_solmix[fi]], dtype=float)
            weights = weights / weights.sum() if weights.sum() else np.array([0.5, 0.5])
            solref = raw.geom_solref[ti] * weights[0] + raw.geom_solref[fi] * weights[1]
            solimp = raw.geom_solimp[ti] * weights[0] + raw.geom_solimp[fi] * weights[1]
            pair_name = f"selftest_contact_{len(pair_names)}"
            ET.SubElement(contact, "pair", {
                "name": pair_name, "geom1": tg.attrib["name"], "geom2": name,
                 "condim": str(max(int(raw.geom_condim[ti]), int(raw.geom_condim[fi]))),
                "friction": words(friction),
                "solref": words(solref),
                "solimp": words(solimp),
            })
            pair_names.append(pair_name)
    visual = xml.find("visual")
    if visual is None:
        visual = ET.SubElement(xml, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "960")
    global_visual.set("offheight", "640")
    modified_xml = ET.tostring(xml, encoding="unicode")
    env.reset_from_xml_string(modified_xml)
    assert np.isclose(env.sim.model.body_mass[body_id], raw.body_mass[body_id] * mass_scale)
    return env, task, modified_xml, pair_names


class Episode:
    def __init__(self, plan):
        self.plan = plan
        self.env, self.task, self.xml, self.pair_names = make_env(
            plan["task_id"], plan["seed"], plan["physics"]
        )
        self.model = self.env.sim.model
        self.target_id = self.env.env.obj_body_id[TARGET]
        self.plate_id = self.env.env.obj_body_id["plate_1"]
        self.site_id = self.env.robots[0].eef_site_id
        self.finger_sides = {
            self.model.geom_name2id(name): side
            for side, key in enumerate(("left_finger", "right_finger"))
            for name in self.env.robots[0].gripper.important_geoms[key]
        }
        self.target_geom_ids = set(np.flatnonzero(
            np.asarray(self.model.geom_bodyid) == self.target_id
        ).tolist())
        self.source = ROOT / "cache/datasets/libero/libero_spatial" / (self.task.name + "_demo.hdf5")
        with h5py.File(self.source) as source:
            demo = source["data"][f"demo_{plan['demo_id']}"]
            self.expert_actions = np.asarray(demo["actions"], dtype=np.float32)
            self.expert_states = np.asarray(demo["states"], dtype=np.float64)
            self.expert_xyz = np.asarray(demo["obs"]["ee_pos"])
            self.expert_rot = Rotation.from_rotvec(np.asarray(demo["obs"]["ee_ori"])).as_matrix()
        self.obs = self.env.set_init_state(self.expert_states[0])
        self.initial_state = self.state()
        self.records = []
        self.actions = []
        self.phases = []
        self.events = []
        self.parameters = self.physics_parameters()
        self.step_index = 0
        self.success = False
        self.capture()

    def state(self):
        return np.asarray(self.env.get_sim_state(), dtype=np.float64).copy()

    def physics_parameters(self):
        return np.concatenate((
            self.model.pair_friction.reshape(-1),
            np.asarray(self.model.body_mass),
            np.asarray(self.model.body_inertia).reshape(-1),
        )).copy()

    def eef(self):
        return np.asarray(self.env.sim.data.site_xpos[self.site_id]).copy()

    def rot(self):
        return np.asarray(self.env.sim.data.site_xmat[self.site_id]).reshape(3, 3).copy()

    def bowl(self):
        return np.asarray(self.env.sim.data.body_xpos[self.target_id]).copy()

    def plate(self):
        return np.asarray(self.env.sim.data.body_xpos[self.plate_id]).copy()

    def relative(self):
        return self.rot().T @ (self.bowl() - self.eef())

    def contacts(self):
        sides = np.zeros(2, dtype=np.int8)
        values = []
        for contact in self.env.sim.data.contact[:self.env.sim.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self.target_geom_ids and b in self.finger_sides:
                sides[self.finger_sides[b]] = 1
                values.append(np.asarray(contact.friction).copy())
            elif b in self.target_geom_ids and a in self.finger_sides:
                sides[self.finger_sides[a]] = 1
                values.append(np.asarray(contact.friction).copy())
        return sides, np.asarray(values, dtype=np.float64).reshape(-1, 5)

    def capture(self):
        sides, friction = self.contacts()
        proprio = np.concatenate((
            np.asarray(self.obs["robot0_eef_pos"]),
            Rotation.from_matrix(self.rot()).as_rotvec(),
            np.asarray(self.obs["robot0_gripper_qpos"]),
        )).astype(np.float32)
        self.records.append({
            "state": self.state(),
            "eef": self.eef(),
            "bowl": self.bowl(),
            "relative": self.relative(),
            "rotation": self.rot(),
            "proprio": proprio,
            "agent_rgb": np.asarray(self.obs["agentview_image"]).copy(),
            "wrist_rgb": np.asarray(self.obs["robot0_eye_in_hand_image"]).copy(),
            "contact_sides": sides,
            "contact_friction": float(friction[:, 0].min()) if len(friction) else -1.0,
            "success": bool(self.env.check_success()),
        })

    def step(self, command, phase):
        command = np.asarray(command, dtype=np.float32)
        assert command.shape == (7,) and np.isfinite(command).all()
        command = np.clip(command, -1.0, 1.0)
        self.actions.append(command)
        self.phases.append(phase)
        clock = float(self.env.sim.data.time)
        self.obs, _, _, _ = self.env.step(command)
        if float(self.env.sim.data.time) <= clock:
            raise RuntimeError("MuJoCo时间回退，物理求解器发生重置，该回合不进入监督数据")
        self.step_index += 1
        if self.step_index % 80 == 0:
            print("PHYSICAL_STEPS", self.step_index, phase, flush=True)
        self.success = bool(self.env.check_success())
        assert np.isfinite(self.state()).all()
        if self.step_index % 50 == 0:
            np.testing.assert_array_equal(self.parameters, self.physics_parameters())
        self.capture()

    def command_to(self, position, orientation, grip):
        controller = self.env.robots[0].controller
        scales = (np.asarray(controller.output_max) - np.asarray(controller.output_min)) / 2
        error_r = Rotation.from_matrix(orientation @ self.rot().T).as_rotvec()
        command = np.r_[(position - self.eef()) / scales[:3], error_r / scales[3:], grip]
        return np.clip(command, -1, 1)

    def move(self, position, orientation=None, grip=1.0, steps=16, phase="adjust"):
        position = np.asarray(position, dtype=np.float64)
        start = self.eef()
        start_r = self.rot()
        target_r = start_r if orientation is None else orientation
        angle = Rotation.from_matrix(target_r @ start_r.T).as_rotvec()
        for n in range(steps):
            u = (n + 1) / steps
            blend = u * u * (3 - 2 * u)
            goal = start + (position - start) * blend
            rot = Rotation.from_rotvec(angle * blend).as_matrix() @ start_r
            self.step(self.command_to(goal, rot, grip), phase)
        return position, target_r

    def hold(self, position, orientation, grip=1.0, steps=8, phase="hold"):
        for _ in range(steps):
            self.step(self.command_to(position, orientation, grip), phase)

    def probe(self, method):
        start = self.step_index
        position, orientation = self.eef(), self.rot()
        before_relative = self.relative()
        if method == "P01":
            end = position + [0, 0, 0.012]
            self.move(end, orientation, steps=10, phase="probe")
            self.hold(end, orientation, steps=8, phase="probe")
        elif method == "P02":
            direction = self.plate() - self.bowl()
            direction[2] = 0
            direction /= max(np.linalg.norm(direction), 1e-8)
            end = position + direction * 0.012
            self.move(end, orientation, steps=10, phase="probe")
            self.hold(end, orientation, steps=8, phase="probe")
        elif method == "P03":
            self.move(position + [0.010, 0, 0], orientation, steps=12, phase="probe")
            self.move(position, orientation, steps=6, phase="probe")
            self.hold(position, orientation, steps=8, phase="probe")
        elif method in ("P04", "P05"):
            axis = np.array([0, 0, 1.0]) if method == "P04" else np.array([1.0, 0, 0])
            turn = Rotation.from_rotvec(axis * np.deg2rad(7)).as_matrix() @ orientation
            self.move(position, turn, steps=10, phase="probe")
            self.hold(position, turn, steps=6, phase="probe")
            self.move(position, orientation, steps=10, phase="probe")
        elif method == "P07":
            self.hold(position, orientation, grip=-0.2, steps=2, phase="probe")
            self.hold(position, orientation, grip=1.0, steps=10, phase="probe")
        else:
            raise ValueError(f"Unsupported executed probe: {method}")
        drift = float(np.linalg.norm(self.relative() - before_relative))
        contact = self.contacts()[0]
        # 真值只用于离线专家分支和记录，导出的策略输入不包含它。
        result = "negative" if drift > 0.005 or not contact.all() else "pass"
        witness = {
            "method": method, "start_step": start, "end_step": self.step_index,
            "result": result, "relative_shift_m": drift,
            "eef_movement_m": float(np.linalg.norm(self.eef() - position)),
            "two_finger_contact_end": bool(contact.all()),
            "task_ended_by_probe": False,
            "decision_source": "offline_privileged_teacher",
        }
        self.events.append(witness)
        return witness

    def regrasp(self, retry=0):
        start = self.step_index
        gi = int(np.flatnonzero(self.expert_actions[:, 6] > 0)[0])
        prep = max(0, gi - 8)
        lifted = np.flatnonzero(self.expert_states[:, 12] - self.expert_states[0, 12] > 0.014)
        finish = max(gi+10, int(lifted[0]))
        translation = self.bowl() - self.expert_states[prep, 10:13]
        if retry:
            translation[1] += 0.006
        orientation = self.expert_rot[prep]
        position = self.eef()
        self.hold(position, self.rot(), grip=-1, steps=8, phase="regrasp")
        safe_height = max(float(position[2] + 0.07), float(self.bowl()[2] + 0.12))
        self.move([*position[:2], safe_height], orientation, grip=-1, steps=20, phase="regrasp")
        approach = self.expert_xyz[prep] + translation
        self.move([*approach[:2], safe_height], orientation, grip=-1, steps=20, phase="regrasp")
        self.move(approach, orientation, grip=-1, steps=16, phase="regrasp")
        for index in range(prep, finish):
            goal = self.expert_xyz[index+1] + translation
            orientation = self.expert_rot[index+1]
            grip = float(self.expert_actions[index,6])
            self.step(self.command_to(goal, orientation, grip), "regrasp")
            self.step(self.command_to(goal, orientation, grip), "regrasp")
        anchor, rotation = self.eef(), self.rot()
        self.hold(anchor, rotation, grip=1, steps=10, phase="regrasp")
        self.events.append({"method": "regrasp", "start_step": start, "end_step": self.step_index,
                            "result": "executed", "retry": retry})

    def finish(self, grasp_offset):
        # 搬运目标在原始任务中已知，末端闭环只跟踪固定的几何路径。
        rotation = self.rot()
        goal = self.plate() + (
            self.expert_states[-1, 10:13] - self.expert_states[-1, 38:41]
        )
        high = max(float(self.eef()[2]), float(goal[2] + grasp_offset[2] + 0.12))
        self.move([*self.eef()[:2], high], rotation, steps=24, phase="transport")
        self.move([*(goal[:2] + grasp_offset[:2]), high], rotation, steps=35, phase="transport")
        end = goal + grasp_offset + [0, 0, 0.007]
        self.move(end, rotation, steps=28, phase="place")
        self.hold(end, rotation, steps=7, phase="place")
        self.hold(end, rotation, grip=-1, steps=8, phase="release")
        self.move(end + [0, 0, 0.060], rotation, grip=-1, steps=18, phase="release")
        self.hold(end + [0, 0, 0.060], rotation, grip=-1, steps=8, phase="release")

    def run(self):
        started = time.monotonic()
        gi = int(np.flatnonzero(self.expert_actions[:, 6] > 0)[0])
        lifted = np.flatnonzero(
            self.expert_states[:, 12] - self.expert_states[0, 12] > 0.014
        )
        fork = max(gi + 8, int(lifted[0]) - 3)
        object_rotation = np.asarray(self.env.sim.data.body_xmat[self.target_id]).reshape(3, 3)
        offset = object_rotation @ np.array([
            float(self.plan["physics"].get("grasp_offset_m", 0)), 0, 0
        ])
        for index, action in enumerate(self.expert_actions[:fork]):
            if np.any(offset) and index >= gi - 10:
                goal = self.expert_xyz[min(index + 1, len(self.expert_xyz)-1)] + offset
                orientation = self.expert_rot[min(index + 1, len(self.expert_rot)-1)]
                action = self.command_to(goal, orientation, float(action[6]))
            self.step(action, "expert_prefix")
        # 固定当前绝对目标，以实际末端反馈制动，避免零指令继续漂移。
        pause_start = self.step_index
        anchor, anchor_r = self.eef(), self.rot()
        self.hold(anchor, anchor_r, steps=12, phase="stabilize")
        self.events.append({
            "method":"stabilize", "start_step":pause_start, "end_step":self.step_index,
            "movement_m": float(np.linalg.norm(self.eef()-anchor)),
            "result":"executed",
        })
        initial_grasp_offset = self.eef() - self.bowl()
        witness = self.probe(self.plan["first_method"])
        if self.plan["feedback"] and witness["result"] == "negative":
            self.regrasp(0)
            witness = self.probe(self.plan.get("second_method", "P01"))
            if witness["result"] == "negative":
                self.regrasp(1)
                witness = self.probe("P01")
            grasp_offset = self.eef() - self.bowl()
        else:
            grasp_offset = initial_grasp_offset
        self.finish(grasp_offset)
        success = bool(self.env.check_success())
        assert len(self.records) == len(self.actions) + 1
        np.testing.assert_array_equal(self.parameters, self.physics_parameters())
        record = {
            "plan": self.plan, "task_name": self.task.name, "language": self.task.language,
            "source_hdf5": str(self.source), "source_demo": self.plan["demo_id"],
            "control_hz": 20, "control_steps": len(self.actions),
            "success": success, "events": self.events,
            "physics_parameters_unchanged": True,
            "teacher": "expert_prefix_and_classical_feedback_controller",
            "policy_deployed": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        return record

    def save(self, directory, metadata):
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "model.xml").write_text(self.xml, encoding="utf-8")
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with h5py.File(directory / "trajectory.hdf5", "w") as target:
            target.attrs["control_hz"] = 20
            target.attrs["task_success"] = metadata["success"]
            target.attrs["policy_deployed"] = False
            target.create_dataset("actions", data=np.asarray(self.actions, dtype=np.float32))
            target.create_dataset("phase", data=np.asarray(self.phases, dtype=h5py.string_dtype("utf-8")))
            for key in self.records[0]:
                values = np.asarray([frame[key] for frame in self.records])
                options = {"compression":"gzip", "compression_opts":1} if key.endswith("_rgb") else {}
                target.create_dataset("observations/" + key, data=values, **options)
            target.create_dataset("initial_state", data=self.initial_state)
        return directory

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    assert plan["first_method"] in SUPPORTED_METHODS
    episode = Episode(plan)
    try:
        metadata = episode.run()
        episode.save(args.output, metadata)
        print("REAL_ROLLOUT", json.dumps(metadata, ensure_ascii=False), flush=True)
    finally:
        episode.close()


if __name__ == "__main__":
    main()
