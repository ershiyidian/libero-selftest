import argparse
import json
import sys
from pathlib import Path

from engine_500 import Episode
from planning_500 import make_plan, load_catalog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_dataset import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--demo", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan = make_plan(args.task, args.demo, 0, 20261010, load_catalog())
    # 验证新增扭转条件，物理摩擦与质量在整个回合中保持固定。
    plan["physics"] = {
        "friction_multiplier": 0.75, "torsional_multiplier": 0.5,
        "mass_multiplier": 1.0, "grasp_offset_m": 0.0,
    }
    plan["condition_ids"] = ["C01", "C02"]
    ep = Episode(plan)
    try:
        meta = ep.run()
        ep.save(args.output / plan["plan_id"], meta)
        quality = audit(args.output / plan["plan_id"] / "trajectory.hdf5", meta)
        result = {"metadata": meta, "audit": quality}
        (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("SCALE_SMOKE", json.dumps(quality, ensure_ascii=False), flush=True)
    finally:
        ep.close()


if __name__ == "__main__":
    main()
