import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SelfTestEpisodes(Dataset):
    def __init__(self, index_path):
        self.index_path = Path(index_path).resolve()
        self.episodes = json.loads(self.index_path.read_text())["episodes"]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        item = self.episodes[index]
        path = self.index_path.parent / item["trajectory"]
        with h5py.File(path) as h5:
            actions = h5["actions"][:]
            phases = h5["phase"].asstr()[:]
            change = np.flatnonzero(phases[1:] != phases[:-1]) + 1
            # 动作块不跨越新的观察或反馈分支，保留实际有效动作数量。
            starts = sorted(set(range(0,len(actions),8)) | set(change.tolist()))
            stops = starts[1:] + [len(actions)]
            targets = np.zeros((len(starts),8,7),dtype=np.float32)
            masks = np.zeros((len(starts),8),dtype=np.bool_)
            previous = np.zeros_like(targets)
            previous_mask = np.zeros_like(masks)
            for k,(start,stop) in enumerate(zip(starts,stops)):
                length=stop-start
                assert 1<=length<=8
                targets[k,:length]=actions[start:stop]
                masks[k,:length]=True
                if k:
                    n=starts[k]-starts[k-1]
                    previous[k,:n]=actions[starts[k-1]:starts[k]]
                    previous_mask[k,:n]=True
            rgb = np.stack([
                h5["observations/agent_rgb"][starts],
                h5["observations/wrist_rgb"][starts]
            ],axis=1).transpose(0,1,4,2,3).copy()
            proprio=h5["observations/proprio"][starts]
        return {
            "images":torch.from_numpy(rgb).float()/255,
            "proprio":torch.from_numpy(proprio),
            "actions":torch.from_numpy(targets),
            "action_mask":torch.from_numpy(masks),
            "previous_executed_actions":torch.from_numpy(previous),
            "previous_action_mask":torch.from_numpy(previous_mask),
            "decision_steps":torch.tensor(starts,dtype=torch.int64),
            "task":item["language"],
            "episode":item["episode"],
        }


def audit(path,metadata):
    with h5py.File(path) as h5:
        actions=h5["actions"][:]
        states=h5["observations/state"][:]
        eef=h5["observations/eef"][:]
        obj=h5["observations/bowl"][:]
        proprio=h5["observations/proprio"][:]
        phases=h5["phase"].asstr()[:]
        assert actions.ndim==2 and actions.shape[1]==7
        count=len(actions)
        assert states.shape[0]==count+1
        assert eef.shape==obj.shape==(count+1,3)
        assert proprio.shape==(count+1,8)
        assert h5["observations/agent_rgb"].shape==(count+1,128,128,3)
        assert h5["observations/wrist_rgb"].shape==(count+1,128,128,3)
        assert all(np.isfinite(array).all() for array in (actions,states,eef,obj,proprio))
        assert np.max(np.abs(actions))<=1.00001
        assert len(phases)==count and "probe" in set(phases)
        np.testing.assert_allclose(np.diff(states[:,0]),0.05,rtol=0,atol=1e-7)
        actual_success=bool(h5["observations/success"][-1])
        assert actual_success==bool(metadata["success"])==bool(h5.attrs["task_success"])
        assert metadata["physics_parameters_unchanged"]
        negative=[e for e in metadata["events"] if e["result"]=="negative"]
        adjustments=[e for e in metadata["events"] if e["method"]=="regrasp"]
        for event in negative:
            assert not event["task_ended_by_probe"]
            assert any(a["start_step"]>=event["end_step"] for a in adjustments) or event is negative[-1]
        object_speed=np.linalg.norm(np.diff(obj,axis=0),axis=1)*20
        eef_speed=np.linalg.norm(np.diff(eef,axis=0),axis=1)*20
        plausible_workspace=bool(np.all(np.abs(obj[:,:2])<1.5) and np.all(obj[:,2]>0.6))
        return {
            "steps":count,"success":actual_success,"negative_tests":len(negative),
            "regrasp_count":len(adjustments),"max_eef_speed_m_s":float(eef_speed.max()),
            "max_object_speed_m_s":float(object_speed.max()),
            "object_stayed_near_workspace":plausible_workspace,
            "all_arrays_consistent":True,"time_continuous":True,
            "eligible_for_success_sft":actual_success and plausible_workspace,
        }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch",type=Path,required=True)
    args=parser.parse_args()
    root=args.batch.resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    report=[]; examples=[]
    for row in manifest["records"]:
        ep=root/row["plan"]["plan_id"]
        values=audit(ep/"trajectory.hdf5",row)
        values["episode"]=ep.name
        report.append(values)
        if values["eligible_for_success_sft"]:
            examples.append({
                "episode":ep.name,"trajectory":str((ep/"trajectory.hdf5").relative_to(root)),
                "language":row["language"],"split_group":row["plan"]["split_group"],
                "source_demo":row["source_demo"],
            })
    assert examples,"没有实际成功的训练样本"
    target=root/"training_index.json"
    target.write_text(json.dumps({"episodes":examples,"policy_inputs":["images","proprio","actions","history"],"hidden_truth_excluded":True},ensure_ascii=False,indent=2))
    data=SelfTestEpisodes(target)
    first=next(iter(DataLoader(data,batch_size=1,shuffle=False,num_workers=0)))
    assert first["actions"].shape[-2:]==(8,7)
    assert first["images"].shape[2:]==(2,3,128,128)
    assert first["proprio"].shape[-1]==8
    total=sum(x["steps"] for x in report)
    summary={
        "physical_episodes":len(report),"successes":sum(x["success"] for x in report),
        "success_training_episodes":len(examples),"physical_steps":total,
        "episodes_with_negative_probe":sum(x["negative_tests"]>0 for x in report),
        "recovered_success_episodes":sum(x["success"] and x["negative_tests"]>0 for x in report),
        "dataloader_checked":True,
        "actual_batch_shapes":{k:list(v.shape) for k,v in first.items() if torch.is_tensor(v)},
        "episodes":report,
    }
    (root/"audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print("DATASET_AUDIT",json.dumps({k:v for k,v in summary.items() if k!="episodes"},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
