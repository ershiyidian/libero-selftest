import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT.parent / 'VLA_SelfTest_ExpertSpine_v41_20261014'
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/VLA-Adapter'), str(EXPERIMENT)]
from scripts.eval_libero_memory import load_policy, get_chunk, process_action
from validate_fixture_pair_v44 import create


def evaluate_one(cfg, vlm, base_head, head, projector, processor, memory_file,
                 folder, mode, max_steps):
    meta = json.loads((folder/'metadata.json').read_text())
    physics = meta['physics']['paired_joint_frictionloss']
    probe = next(e for e in meta['events'] if e.get('method') == meta['method'])
    split = int(probe['end_step'])
    with h5py.File(folder/'trajectory.hdf5') as data:
        commands = np.asarray(data['actions'], dtype=np.float32)
    ep = create(meta['task_id'], int(meta['demo_id']), physics, record_images=True)
    started = time.perf_counter()
    policy = base_head if mode == 'pretrained' else head
    try:
        # 测试从实际源动作推进到探针末端；此后全部执行模型输出。
        ep.fixture_shield = None
        head.reset_memory()
        last_observation = None
        last_executed = []
        for index in range(split):
            if index == max(0, split - 8) and mode in ('trained_history', 'trained_window'):
                observation = ep.env.env._get_observations(force_update=True)
                last_observation = copy.deepcopy(observation)
                if mode == 'trained_history':
                    get_chunk(cfg, vlm, observation, ep.task.language, processor, head, projector)
            ep.execute(commands[index], 'probe' if index >= int(probe['start_step']) else 'expert_prefix')
            if index >= split - 8:
                last_executed.append(commands[index].copy())
                if mode == 'trained_history':
                    head.record_executed_action(commands[index])
        history = []
        first_prediction = None
        success = False
        for decision in range((max_steps + 7) // 8):
            if mode == 'trained_no_history':
                head.reset_memory()
            if mode == 'trained_window':
                # 与两帧训练样本对齐：前一真实观察和实际执行的动作块。
                head.reset_memory()
                get_chunk(cfg, vlm, last_observation, ep.task.language, processor, head, projector)
                for action in last_executed:
                    head.record_executed_action(action)
            obs = ep.env.env._get_observations(force_update=True)
            chunk, _ = get_chunk(cfg, vlm, obs, ep.task.language, processor, policy, projector)
            last_observation = copy.deepcopy(obs)
            last_executed = []
            if first_prediction is None:
                first_prediction = [process_action(a.copy(), cfg.model_family).tolist() for a in chunk]
            for normalized in chunk:
                command = process_action(normalized.copy(), cfg.model_family)
                ep.execute(command, 'learned_policy')
                head.record_executed_action(command)
                last_executed.append(np.asarray(command).copy())
                history.append({'action': np.asarray(command).tolist(),
                                'joint': ep.joint_pos(), 'eef': ep.eef().tolist(),
                                'goal': bool(ep.env.check_success())})
                if ep.env.check_success():
                    success = True
                    break
                if len(history) >= max_steps:
                    break
            if success or len(history) >= max_steps:
                break
        first = np.asarray(first_prediction)
        target = commands[split:split+8]
        return {'source': str(folder), 'task': ep.task_id, 'world_frictionloss': physics,
                'mode': mode, 'goal_success': success, 'learned_steps': len(history),
                'teacher_prefix_steps': split, 'first_chunk_physical_l1': float(np.abs(first-target).mean()),
                'first_chunk_predicted': first_prediction, 'first_chunk_teacher': target.tolist(),
                'final_joint': ep.joint_pos(), 'policy_history': history,
                'seconds': time.perf_counter() - started,
                'scope': 'Actual learned control after an expert-executed microprobe; not a whole-task policy benchmark',
                'full_safety_audit_run': False, 'pretrained_checkpoint': 'spatial_pro',
                'memory_checkpoint': str(memory_file) if mode != 'pretrained' else None}
    finally:
        ep.close()
        head.reset_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pair-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, default=96)
    args = parser.parse_args()
    args.checkpoint, args.pair_dir, args.output = (p.resolve() for p in
        (args.checkpoint, args.pair_dir, args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(45)
    cfg, vlm, base, head, projector, processor = load_policy(ROOT/'checkpoints/spatial_pro')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['memory_state_dict']
    result = head.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(k.startswith(('memory_update.', 'memory_readers.')) for k in result.missing_keys):
        raise RuntimeError('已训练的记忆权重没有完整载入')
    rows = []
    with torch.inference_mode():
        for mode in ('pretrained', 'trained_history', 'trained_window', 'trained_no_history'):
            for world in ('normal', 'risk_adapted'):
                row = evaluate_one(cfg, vlm, base, head, projector, processor, args.checkpoint,
                                   args.pair_dir/world, mode, args.max_steps)
                rows.append(row)
                (args.output/'rollouts.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                print('ACTIVE_POLICY_ROLLOUT', json.dumps({k: v for k, v in row.items()
                    if k not in ('policy_history', 'first_chunk_predicted', 'first_chunk_teacher')}, ensure_ascii=False), flush=True)
    summary = {mode: {'success': sum(r['goal_success'] for r in rows if r['mode'] == mode),
                      'trials': sum(r['mode'] == mode for r in rows)}
               for mode in ('pretrained', 'trained_history', 'trained_window', 'trained_no_history')}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('ACTIVE_POLICY_SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
