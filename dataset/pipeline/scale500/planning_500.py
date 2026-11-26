import random
from pathlib import Path

import h5py
import yaml


DATA = Path(__file__).resolve().parents[2]
METHODS = ("P01", "P02", "P03", "P04", "P05", "P07")
CONDITIONS = ("C01", "C02", "C03", "C04")
PHYSICS_KEYS = {
    "C01": "friction_multiplier",
    "C02": "torsional_multiplier",
    "C03": "mass_multiplier",
    "C04": "grasp_offset_m",
}
METHOD_NAMES = {
    "P01": "轻抬并保持", "P02": "沿搬运方向侧移",
    "P03": "两种速度试运", "P04": "转腕并返回",
    "P05": "小幅倾斜", "P07": "部分松开再闭合",
}
CONDITION_NAMES = {
    "C01": "抓持滑动摩擦降低", "C02": "抓持扭转摩擦降低",
    "C03": "物体质量增加", "C04": "初次抓点偏移",
}


def source_inventory(lab_root: Path) -> dict:
    source_dir = lab_root / "cache/datasets/libero/libero_spatial"
    # 任务编号以LIBERO原生任务顺序为准，由工作进程实际核对。
    result = {}
    for path in sorted(source_dir.glob("*_demo.hdf5")):
        with h5py.File(path) as file:
            demos = [
                int(name.rsplit("_", 1)[-1]) for name, group in file["data"].items()
                if len(group["actions"]) >= 40 and bool(group["rewards"][-1] > 0)
            ]
        result[path.name] = sorted(demos)
    assert len(result) == 10
    assert all(values for values in result.values())
    return result


def make_plan(task: int, demo: int, attempt: int, seed: int, catalog: dict) -> dict:
    rng = random.Random(seed + 104729 * attempt)
    public_rng = random.Random(seed + 1009 * task + 17 * demo)
    first = METHODS[(task * 50 + demo) % len(METHODS)]
    allowed = [
        cid for cid in CONDITIONS
        if first in catalog["conditions"][cid]["methods"]
    ]
    size = rng.choices((1, 2, 3), weights=(50, 40, 10))[0]
    chosen = sorted(rng.sample(allowed, min(size, len(allowed))))
    physics = {
        "friction_multiplier": 1.0, "torsional_multiplier": 1.0,
        "mass_multiplier": 1.0, "grasp_offset_m": 0.0,
    }
    for cid in chosen:
        nominal = 0.0 if cid == "C04" else 1.0
        values = [
            x for x in catalog["conditions"][cid]["candidates"]
            if x != nominal
        ]
        physics[PHYSICS_KEYS[cid]] = rng.choice(values)

    parameters = {}
    for method in METHODS:
        parameters[method] = {
            "distance_m": public_rng.choice((0.006, 0.009, 0.012, 0.015)),
            "angle_deg": public_rng.choice((3.0, 5.0, 7.0, 9.0)),
            "movement_steps": public_rng.choice((8, 10, 12, 16)),
            "hold_steps": public_rng.choice((6, 8, 10, 12)),
            "direction_sign": public_rng.choice((-1, 1)),
            "opening_steps": public_rng.choice((1, 2, 3)),
        }
    second = METHODS[(METHODS.index(first) + 2) % len(METHODS)]
    # 同一个真实源历史共享首测、测试幅度和检查阶段，隐藏参数不参与这些选择。
    public_sequence = [first, second] if demo % 5 == 0 else [first]
    stage = "transport_check" if demo % 7 == 0 else "grasp_check"
    return {
        "plan_id": f"episode_{attempt:06d}",
        "task_id": task, "demo_id": demo,
        "seed": seed + task * 1009 + demo * 17,
        "physics": physics, "condition_ids": chosen,
        "first_method": first, "second_method": second,
        "candidate_methods": [first, second, "P01"],
        "public_test_sequence": public_sequence,
        "probe_parameters": parameters,
        "feedback": True, "public_stage": stage,
        "source_kind": "original_libero_expert",
        "split_group": f"spatial_{task}_demo_{demo}",
        "split": ("test" if demo % 10 == 9 else "validation" if demo % 10 == 8 else "train"),
        "planner_source": "validated_condition_method_pool",
    }


def load_catalog() -> dict:
    return yaml.safe_load((DATA / "pools/catalog.yaml").read_text(encoding="utf-8"))
