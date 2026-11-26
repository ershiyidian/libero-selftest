import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/VLA-Adapter'),str(ROOT/'official_training')]
from scripts.train_memory_quick_sft import capture_two_real_vlm_features
from scripts.eval_libero_memory import load_policy
from selftest.losses import masked_action_l1_loss


def samples_from_manifest(manifest, limit):
    records=json.loads(Path(manifest).read_text())
    result=[]
    for row in records:
        folder=Path(row['source_path'])
        meta=json.loads((folder/'metadata.json').read_text())
        roles=json.loads((folder/'training_roles.json').read_text())
        if not (row.get('accepted') and row.get('replay_passed')):
            raise ValueError('训练样本必须已经通过真实仿真及独立重放')
        probes=sorted((x for x in meta['events'] if x.get('method')==meta['method']),key=lambda x:x['start_step'])
        if not probes:continue
        with h5py.File(folder/'trajectory.hdf5') as h:
            actions=np.asarray(h['actions'],dtype=np.float32)
            mask=np.ones(len(actions),dtype=np.float32)
            for first,last in roles['context_only_ranges']:mask[first:last]=0
            if roles['action_count']!=len(actions):raise ValueError('动作与掩码时间轴错位')
            pivot=min(max(8,int(row.get('decision_step',probes[0]['end_step']))),len(actions)-8)
            start=pivot-8
            if len(actions)<16 or mask[start:start+16].sum()==0:continue
            # 生成器保存的是MuJoCo原始相机方向；官方LIBERO推理旋转180度。
            images=[(np.asarray(h['observations/agent_rgb'][i])[::-1,::-1].copy(),
                     np.asarray(h['observations/wrist_rgb'][i])[::-1,::-1].copy())
                    for i in (start,pivot)]
            result.append({'source':str(folder),'task':meta['task_language'],
                           'paired_audit':row.get('audit_path') if row.get('audit_protocol')=='observable_stiction_duration_v44' else None,
                           'times':[start,pivot],'images':images,
                           'proprio':[np.asarray(h['observations/proprio'][i],dtype=np.float32) for i in (start,pivot)],
                           'actions':[actions[start:start+8],actions[pivot:pivot+8]],
                           'loss_mask':[mask[start:start+8],mask[pivot:pivot+8]]})
        if len(result)>=limit:break
    if len(result)<2:raise ValueError('至少需要两个真实时序窗口')
    return result


def tensor(value, device):
    return torch.as_tensor(value,device=device,dtype=torch.bfloat16)


def sequence_loss(head,projector,item,device,previous=None):
    head.reset_memory()
    losses=[]
    actual=item['observations']
    history_source=actual[0] if previous is None else previous['observations'][0]
    for step,frame in enumerate((history_source,actual[1])):
        pred=head.predict_action(tensor(frame['features'],device),tensor(frame['proprio'],device),projector)
        target=tensor(frame['target'],device).unsqueeze(0)
        weight=torch.as_tensor(item['masks'][step],device=device,dtype=torch.float32).unsqueeze(0)
        if previous is None and weight.sum()>0 or step==1 and weight.sum()>0:
            losses.append(masked_action_l1_loss(pred.float(),target.float(),weight))
        if step==0:
            for a in history_source['executed']:head.record_executed_action(a)
    if not losses:raise ValueError('无监督动作的窗口被采样')
    return torch.stack(losses).mean()


def evaluate(head,projector,cache,device):
    head.eval()
    with torch.no_grad():
        full=[float(sequence_loss(head,projector,item,device)) for item in cache]
        paired=[]
        mismatched=[]
        for i,item in enumerate(cache):
            # 对两个条件均只比较第二次预测，当前图像和目标完全不变。
            isolated={**item,'masks':[np.zeros(8,dtype=np.float32),item['masks'][1]]}
            paired.append(float(sequence_loss(head,projector,isolated,device)))
            mismatched.append(float(sequence_loss(head,projector,isolated,device,cache[(i+1)%len(cache)])))
    head.reset_memory()
    return {'mean_masked_l1':float(np.mean(full)),
            'current_action_l1_correct_history':float(np.mean(paired)),
            'current_action_l1_mismatched_history':float(np.mean(mismatched)),
            'per_pair':full}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--pairs',type=int,default=6)
    ap.add_argument('--updates',type=int,default=64)
    args=ap.parse_args()
    args.manifest=args.manifest.resolve()
    args.output=args.output.resolve()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    random.seed(44);np.random.seed(44);torch.manual_seed(44);torch.set_num_threads(3)
    begin=time.perf_counter()
    samples=samples_from_manifest(args.manifest,args.pairs)
    cfg,vlm,base_head,head,projector,processor=load_policy(ROOT/'checkpoints/spatial_pro')
    statistics=vlm.norm_stats[cfg.unnorm_key]['action']
    cache=[]
    for sample in samples:
        item=capture_two_real_vlm_features(cfg,vlm,base_head,projector,processor,sample,statistics)
        item['masks']=sample['loss_mask']
        cache.append(item)
        print('AUTHENTIC_SELFTEST_FEATURES',item['source'],item['times'],[m.tolist() for m in item['masks']],flush=True)
    del vlm,base_head,processor
    torch.cuda.empty_cache()
    device=next(head.parameters()).device
    for p in head.parameters():p.requires_grad_(False)
    for p in projector.parameters():p.requires_grad_(False)
    gates=[];memory=[]
    for name,p in head.named_parameters():
        if name.startswith(('memory_readers.','memory_update.')):
            p.requires_grad_(True)
            (gates if name.endswith('.gate') else memory).append(p)
    before=evaluate(head,projector,cache,device)
    optimizer=torch.optim.AdamW([{'params':memory,'lr':0.0001},{'params':gates,'lr':0.0005}],weight_decay=0.)
    logs=[]
    for update in range(args.updates):
        head.train();optimizer.zero_grad(set_to_none=True)
        item=cache[update%len(cache)]
        loss=sequence_loss(head,projector,item,device)
        loss.backward()
        g=lambda ps:sum(float(p.grad.float().abs().sum()) for p in ps if p.grad is not None)
        mg,gg=g(memory),g(gates)
        torch.nn.utils.clip_grad_norm_(memory+gates,1.)
        optimizer.step();head.reset_memory()
        row={'update':update+1,'loss':float(loss.detach()),'memory_grad_l1':mg,'gate_grad_l1':gg,
             'gates':[float(p.detach()) for p in gates]}
        logs.append(row)
        if update%8==0 or update+1==args.updates:print('MASKED_PRETRAINED_SFT',json.dumps(row),flush=True)
    after=evaluate(head,projector,cache,device)
    weight={n:p.detach().cpu() for n,p in head.state_dict().items() if n.startswith(('memory_update.','memory_readers.'))}
    torch.save({'memory_state_dict':weight,'pretrained_policy':'VLA-Adapter spatial_pro',
                'action_statistics':statistics,'source_manifest':str(args.manifest)},args.output/'memory_only.pt')
    report={'source_pairs':len(cache),'updates':args.updates,'before':before,'after':after,
            'nonzero_memory_grad_updates':sum(r['memory_grad_l1']>0 for r in logs),
            'nonzero_gate_grad_updates':sum(r['gate_grad_l1']>0 for r in logs),
            'only_memory_parameters_trained':True,'pretrained_action_head_loaded':True,
            'raw_action_history_preserved':True,'target_normalization':'official spatial_pro quantiles, gripper inverted to 0-close/1-open',
            'image_orientation':'MuJoCo raw RGB rotated 180 degrees, same as official LIBERO inference',
            'actual_saved_loss_masks_used':True,
            'G7_training_pairs':len({x['paired_audit'] for x in samples if x.get('paired_audit')}),
            'source_physical_trajectories':len({x['source'] for x in samples}),
            'curriculum':'paired active traces' if any(x.get('paired_audit') for x in samples) else 'accepted recovery demonstrations',
            'elapsed_seconds':time.perf_counter()-begin,'training_log':logs}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('MASKED_PRETRAINED_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='training_log'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
