"""Compare matched 10x16 ordinary rollouts and Z-1 forked rollouts at aligned decision times."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CHUNKS_AFTER_FORK = (0, 1, 2, 4, 8)
FIELDS = ("first_action", "action_chunk", "proprio", "history", "camera", "gripper")


def sample_at(record, index):
    if len(record["samples"]) <= index:
        return None
    s = record["samples"][index]
    action = s["actions_normalized"].float().numpy()
    pixel = s["pixel_values"].float()
    return {
        "first_action": action[0, :6],
        "action_chunk": action[:, :6].reshape(-1),
        "gripper": action[:, 6],
        "proprio": s["proprio"].float().numpy().reshape(-1),
        "history": s["previous_actions"].float().numpy().reshape(-1),
        "camera": F.adaptive_avg_pool2d(pixel, (12, 12)).numpy().reshape(-1),
        "raw_pixels": pixel,
    }


def pairwise_rms(array):
    if len(array) < 2:
        return None
    differences = array[:, None] - array[None, :]
    return float(np.sqrt(np.square(differences).mean(axis=-1))[
        np.triu_indices(len(array), k=1)
    ].mean())


def summarize_group(records, fork_index, offsets):
    labels = np.asarray([int(r["reward"]) for r in records])
    per_time = {}
    for k in offsets:
        aligned = [(i, sample_at(r, fork_index+k)) for i, r in enumerate(records)]
        alive = [(i, s) for i, s in aligned if s is not None]
        d = {"n_alive": len(alive)}
        if len(alive) >= 2:
            for field in FIELDS:
                matrix = np.stack([s[field] for _, s in alive]).astype(np.float32)
                if field == "gripper":
                    disagreement = (matrix[:,None] != matrix[None,:]).mean(-1)
                    d["gripper_disagreement"] = float(disagreement[np.triu_indices(len(matrix),1)].mean())
                else:
                    d[field+"_pair_rms"] = pairwise_rms(matrix)
            motions = np.stack([s["first_action"] for _,s in alive])
            y = labels[[i for i,_ in alive]]
            diff = np.sqrt(((motions[:,None,:]-motions[None,:,:])**2).mean(-1))
            cross,within=[],[]
            for i in range(len(y)):
                for j in range(i+1,len(y)):
                    (cross if y[i]!=y[j] else within).append(diff[i,j])
            d["reward_conditioned_action_separation"] = float(np.mean(cross)-np.mean(within)) if cross and within else None
        per_time[str(k)] = d
    return per_time


def data_check(regular, forked, task, trial, base_checkpoint, fork_size):
    for label,record in (("ordinary",regular),("branch",forked)):
        if record["task_id"] != task or record["trial"] != trial:
            raise ValueError(f"Wrong task/trial in {label} trajectory")
        if record["checkpoint"] != base_checkpoint:
            raise ValueError("Comparison must use an identical policy checkpoint")
        for key,expected in (("sigma",0.1),("exploration_mode","temporally_correlated"),
                             ("temperature",1.0),("rho",0.8)):
            if record.get(key) != expected:
                raise ValueError(f"Inconsistent exploration parameter {key}")
    if not forked["prefix_masked_from_grpo"] or forked["shared_prefix_chunks"] != fork_size:
        raise ValueError("Z-1 samples must contain only post-fork trainable action chunks")


def compare(baseline_path, branch_path, output_path, task_ids, fork_chunks=8):
    previous_summary = json.loads((baseline_path/"rollout_report.json").read_text())
    after_summary = json.loads((branch_path/"rollout_report.json").read_text())
    base_checkpoint=previous_summary["episodes"][0]["checkpoint"]
    aggregate = {
        "reference_checkpoint":base_checkpoint,
        "randomness":"sigma=0.1, temperature=1.0, rho=0.8",
        "fork_control_step":fork_chunks*8,
        "definition":"Independent trajectory aligned at decision index fork_chunks; cloned trajectory begins there. At k=0, Z-1 has one identical observation but different sampled actions.",
        "by_task":[],"grouped":{},
    }
    for task in task_ids:
        ordinary = [torch.load(baseline_path/f"task{task:02d}_trial{trial:02d}.pt",map_location="cpu",weights_only=True) for trial in range(16)]
        forked = [torch.load(branch_path/f"task{task:02d}_trial{trial:02d}.pt",map_location="cpu",weights_only=True) for trial in range(16)]
        for trial,(a,b) in enumerate(zip(ordinary,forked)):
            data_check(a,b,task,trial,base_checkpoint,fork_chunks)
        per_ordinary=summarize_group(ordinary,fork_chunks,CHUNKS_AFTER_FORK)
        per_branch=summarize_group(forked,0,CHUNKS_AFTER_FORK)
        first_obs=[sample_at(r,0) for r in forked]
        reference=first_obs[0]
        errors={
            "proprio_max_abs":max(float(np.max(np.abs(s["proprio"]-reference["proprio"]))) for s in first_obs),
            "history_max_abs":max(float(np.max(np.abs(s["history"]-reference["history"]))) for s in first_obs),
            "camera_max_abs":max(float((s["raw_pixels"]-reference["raw_pixels"]).abs().max()) for s in first_obs),
        }
        entry={
            "task_id":task,
            "ordinary_success":sum(r["reward"] for r in ordinary),
            "branch_success":sum(r["reward"] for r in forked),
            "ordinary_chunks":sum(r["full_action_chunks"] for r in ordinary),
            "branch_trainable_suffix_chunks":sum(r["full_action_chunks"] for r in forked),
            "fork_input_exact":errors,
            "ordinary":per_ordinary,
            "branch":per_branch,
        }
        aggregate["by_task"].append(entry)
        print("Z1_DIVERSITY_TASK",json.dumps({
            "task":task,"ordinary_success":entry["ordinary_success"],
            "branch_success":entry["branch_success"],
            "fork_input_exact":errors,
            "ordinary_fork_action_rms":per_ordinary['0']['action_chunk_pair_rms'],
            "branch_fork_action_rms":per_branch['0']['action_chunk_pair_rms'],
            "ordinary_fork_proprio_rms":per_ordinary['0']['proprio_pair_rms'],
            "branch_fork_proprio_rms":per_branch['0']['proprio_pair_rms'],
        }),flush=True)
        del ordinary,forked

    metrics=('first_action_pair_rms','action_chunk_pair_rms','proprio_pair_rms',
             'history_pair_rms','camera_pair_rms','gripper_disagreement',
             'reward_conditioned_action_separation')
    for mode in ("ordinary","branch"):
        aggregate["grouped"][mode]={}
        for k in CHUNKS_AFTER_FORK:
            data = [entry[mode][str(k)] for entry in aggregate['by_task']]
            aggregate['grouped'][mode][str(k)] = {
                'mean_n_alive':float(np.mean([row['n_alive'] for row in data])),
                **{metric:float(np.mean([row[metric] for row in data if row.get(metric) is not None]))
                   for metric in metrics if any(row.get(metric) is not None for row in data)}
            }
    for mode in ("ordinary","branch"):
        reward=sum(entry[f'{mode}_success'] for entry in aggregate['by_task'])
        aggregate[mode]={
            'trials':len(task_ids)*16,'successes':reward,
            'success_rate':reward/(len(task_ids)*16),
            'mixed_reward_groups':sum(0<entry[f'{mode}_success']<16 for entry in aggregate['by_task']),
            'action_chunks':sum(entry['ordinary_chunks' if mode=='ordinary' else 'branch_trainable_suffix_chunks'] for entry in aggregate['by_task']),
            'rollout_wall_seconds':previous_summary['duration_seconds'] if mode=='ordinary' else after_summary['duration_seconds'],
        }

    delta=np.array([(e['branch_success']-e['ordinary_success'])/16 for e in aggregate['by_task']])
    rng=np.random.default_rng(20261012)
    boot=rng.choice(delta,size=(12000,len(delta)),replace=True).mean(axis=1)
    aggregate['paired_task_success_delta']={
        'mean_points':float(delta.mean()*100),
        'bootstrap_95pct_interval_points':[float(x*100) for x in np.quantile(boot,[0.025,0.975])],
        'number_of_tasks':len(delta),
    }
    output_path.mkdir(parents=True,exist_ok=True)
    (output_path/'comparison.json').write_text(json.dumps(aggregate,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric,title,file in (
        ('action_chunk_pair_rms','Action-chunk diversity after common fork','action_diversity.png'),
        ('proprio_pair_rms','Robot state diversity after common fork','state_diversity.png'),
        ('camera_pair_rms','Vision input diversity after common fork','visual_diversity.png'),
    ):
        fig,ax=plt.subplots(figsize=(6.6,4.1))
        for mode,label in (('ordinary','Independent rollouts'),('branch','Shared-prefix branch')):
            timeline=aggregate['grouped'][mode]
            ax.plot(CHUNKS_AFTER_FORK,[timeline[str(k)][metric] for k in CHUNKS_AFTER_FORK],
                    marker='o',linewidth=2,label=label)
        ax.set(xlabel='Decision chunks after t=64 (each = 8 controls)',ylabel='Mean pairwise RMS',title=title)
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output_path/file,dpi=175)
        plt.close(fig)
    print('Z1_DIVERSITY_FINAL',json.dumps({
        'ordinary':aggregate['ordinary'],'branch':aggregate['branch'],
        'task_success_delta':aggregate['paired_task_success_delta'],
        'at_fork_ordinary':aggregate['grouped']['ordinary']['0'],
        'at_fork_branch':aggregate['grouped']['branch']['0'],
        'after_4_ordinary':aggregate['grouped']['ordinary']['4'],
        'after_4_branch':aggregate['grouped']['branch']['4'],
    }),flush=True)
    return aggregate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ordinary',type=Path,required=True)
    parser.add_argument('--branch',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fork-chunks',type=int,default=8)
    parser.add_argument('--tasks',nargs='+',type=int,default=list(range(10)))
    args=parser.parse_args()
    compare(args.ordinary.resolve(),args.branch.resolve(),args.output.resolve(),args.tasks,args.fork_chunks)


if __name__=='__main__':
    main()
