import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from engine import Episode


def verify_episode(directory):
    metadata=json.loads((directory/"metadata.json").read_text())
    with h5py.File(directory/"trajectory.hdf5") as h5:
        actions=h5["actions"][:]
        expected=h5["observations/state"][:]
    episode=Episode(metadata["plan"])
    maximum=0.0
    try:
        np.testing.assert_allclose(episode.state(),expected[0],rtol=0,atol=1e-8)
        for i,action in enumerate(actions):
            episode.step(action,"replay")
            maximum=max(maximum,float(np.max(np.abs(episode.state()-expected[i+1]))))
        actual_success=bool(episode.env.check_success())
        assert actual_success==metadata["success"]
        assert maximum<1e-4,f"重放偏差过大：{maximum}"
        result={
            "episode":directory.name,"actions_replayed":len(actions),
            "max_full_state_absolute_error":maximum,"success":actual_success,
            "physical_replay":True,
        }
    finally:
        episode.close()
    print("VERIFIED_PHYSICAL_REPLAY",json.dumps(result),flush=True)
    return result


def run_control(directory,output):
    original=json.loads((directory/"metadata.json").read_text())
    plan={**original["plan"],"feedback":False,"plan_id":original["plan"]["plan_id"]+"_ignore_feedback"}
    episode=Episode(plan)
    try:
        metadata=episode.run()
        episode.save(output,metadata)
    finally:
        episode.close()
    end=next(e["end_step"] for e in original["events"] if e["method"].startswith("P"))
    with h5py.File(directory/"trajectory.hdf5") as a,h5py.File(output/"trajectory.hdf5") as b:
        np.testing.assert_array_equal(a["actions"][:end],b["actions"][:end])
        error=float(np.max(np.abs(a["observations/state"][:end+1]-b["observations/state"][:end+1])))
        assert error<1e-5
    result={
        "episode":directory.name,"identical_actions_through_probe":True,
        "max_state_difference_through_probe":error,
        "with_feedback_success":original["success"],"without_feedback_success":metadata["success"],
        "physics_identical":original["plan"]["physics"]==metadata["plan"]["physics"],
        "warning":"单个初态的机制检查，不是总体统计结论",
    }
    print("PAIRED_FEEDBACK_CONTROL",json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",type=Path,required=True)
    parser.add_argument("--verify",nargs="+",default=["selftest_000","selftest_003"])
    parser.add_argument("--control",default="selftest_003")
    args=parser.parse_args()
    results=[verify_episode(args.batch/n) for n in args.verify]
    paired=run_control(args.batch/args.control,args.batch/(args.control+"_ignore_feedback"))
    output={"replays":results,"paired_control":paired}
    (args.batch/"replay_verification.json").write_text(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
