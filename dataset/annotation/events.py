import base64
import json
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_RELEASE = Path('/data/guest/vla_branch_lab/VLA_SelfTest_Curated130_v46_20261015/outputs/curated130_exact8_final_v56')


@dataclass
class Event:
    event_id: str
    episode_id: str
    task_id: str
    relative_path: str
    kind: str
    step: int
    probe_index: int
    start: int
    end: int
    return_end: int
    frame_steps: list[int]
    next_phase: str
    planned_motion: str
    probe_focus: str
    planned_controls: list
    oracle: dict
    action_loss_enabled: bool

    def public(self):
        # 事件文件名含probe_success/failed_recovery，不能传给语言教师暗示未来结果。
        return {'kind': self.kind, 'decision_step': self.step,
                'frame_steps': self.frame_steps, 'probe_start': self.start,
                'probe_end': self.end if self.kind == 'feedback' else None,
                'next_phase': self.next_phase, 'planned_motion': self.planned_motion,
                'probe_focus': self.probe_focus,
                'planned_control_mean_xyz': np.round(np.mean(np.asarray(self.planned_controls)[:, :3], axis=0), 4).tolist()}


PHASE_INTENTS = {
    'improve': 'Adjust gripper contact with the control and retry the operation',
    'recontact': 'Return near the target and realign the gripper with the contact point',
    'grasp_correction': 'Readjust the grasp position and close the gripper',
    'fixture_correction': 'Readjust contact with the drawer or door',
    'restore_expert_waypoint': 'Move to a suitable pose and continue the task',
    'planned_transport': 'Continue carrying the currently held object',
    'planned_approach': 'Approach the placement target and adjust the arm pose',
    'expert_suffix': 'Continue the current object manipulation; do not claim task completion prematurely',
    'corrective_push': 'Reestablish edge contact and keep pushing the target',
    'probe_return': 'Retract from the small test motion',
    'support_seek': 'Approach the supporting surface slowly',
    'supported_release': 'Release the object near the supporting surface',
    'task_fixture': 'Continue operating the drawer or door',
    'withdraw': 'Move the arm away from the target',
    'final_hold': 'Hold the current pose briefly',
    'final_settle': 'Hold the current pose briefly',
}


def read_selection(root: Path) -> list[dict]:
    rows = json.loads((root / 'selection.json').read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('selection.json必须包含非空轨迹列表')
    seen = set()
    for row in rows:
        identity = (row['task_id'], row['demo'])
        if identity in seen:
            raise ValueError(f'重复的任务/专家轨迹: {identity}')
        seen.add(identity)
        for key in ('trajectory_file', 'loss_mask_file', 'quality_audit_file'):
            path = (root / row[key]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f'最终清单文件缺失或路径越界: {key} {path}')
    return rows


def normalized_probes(meta: dict) -> list[dict]:
    result = []
    source = meta.get('probes')
    if source is None:
        source = [x for x in meta.get('events', [])
                  if x.get('result') in ('negative', 'pass')]
    for index, value in enumerate(source):
        probe = dict(value)
        probe['outcome'] = ('pass' if value.get('passed') else 'negative') if 'passed' in value else value['result']
        probe['start'] = int(value['start_step'])
        probe['end'] = int(value['end_step'])
        probe['return_end'] = int(value.get('return_end_step', probe['end']))
        probe['index'] = index
        if not 0 <= probe['start'] < probe['end'] <= probe['return_end']:
            raise ValueError(f'试探时间顺序异常: {probe}')
        result.append(probe)
    result.sort(key=lambda x: x['start'])
    if not result:
        raise ValueError('已验收的自测轨迹没有可解析的试探事件')
    return result


def motion_description(probe: dict, meta: dict, action_slice: np.ndarray) -> str:
    # 这些信息只描述示范下一步的动作意图，不向语言教师透露执行后的物理结果。
    name = probe.get('executed_probe_action', '')
    if name == 'wrist_view_change':
        return 'Rotate the wrist a little to inspect current contact from another angle'
    if name == 'short_wrist_turn':
        return 'Turn the wrist slightly to check whether the grasp holds'
    if name == 'native_expert_initial_lift' or 'hand_displacement' in probe:
        return 'Lift the hand slightly to check whether the object follows'
    if probe.get('probe_mechanism') == 'knob_contact':
        return 'Test the stove control with a small motion; do not assume the burner switched on or off'
    if probe.get('kind') == 'drawer' or 'fixture_joint_name' in probe or 'joint_response' in probe:
        return 'Push or pull the drawer slightly in the intended direction to check engagement'
    if 'hinge_response_rad' in probe or name == 'door_arc':
        return 'Pull or push slightly along the door arc to check whether it moves'
    if 'object_motion_m' in probe or probe.get('kind') == 'push':
        return 'Push the target a small distance in the task direction to check whether it moves'
    if name in ('short_contact_translation', 'source_actuator_segment', 'short_task_direction_motion'):
        return 'Try a short contact movement in the intended task direction'
    movement = np.mean(np.abs(action_slice[:, :6]), axis=0)
    if movement[2] >= max(movement[0], movement[1]) and np.mean(action_slice[:, 2]) > 0:
        return 'Raise the arm slightly to test contact'
    if movement[3:].sum() > movement[:3].sum():
        return 'Adjust wrist orientation slightly to test contact'
    return 'Make a small contact test, then decide based on what is observed'


def oracle_summary(probe: dict) -> dict:
    result = {'outcome': probe['outcome'], 'start': probe['start'], 'end': probe['end'],
              'return_end': probe['return_end']}
    for key in ('actual_motion', 'object_motion_m', 'hand_motion_m', 'joint_response',
                'hinge_response_rad', 'hand_displacement', 'object_displacement',
                'two_finger_contact_end', 'finger_sides', 'object_relative_shift_m',
                'max_object_relative_shift_m', 'object_relative_rotation_rad',
                'eef_displacement_m', 'actual_object_tilt_deg', 'kind'):
        if key in probe:
            result[key] = probe[key]
    return result


def probe_focus(probe: dict) -> str:
    if probe.get('kind') == 'support' or 'hand_displacement' in probe:
        return 'grasped_object'
    if probe.get('kind') == 'drawer' or 'joint_response' in probe:
        return 'drawer'
    if probe.get('executed_probe_action') == 'door_arc' or 'hinge_response_rad' in probe:
        return 'door'
    if probe.get('kind') == 'push' or 'object_motion_m' in probe:
        return 'pushed_object'
    if probe.get('kind') == 'switch' or probe.get('probe_mechanism') == 'knob_contact':
        return 'switch'
    return 'contact_target'


def episode_events(root: Path, row: dict) -> tuple[dict, list[Event], list[dict]]:
    folder = root / row['relative_path']
    meta = json.loads((folder / 'metadata.json').read_text())
    mask = np.load(root / row['loss_mask_file'], allow_pickle=False)
    probes = normalized_probes(meta)
    events, omitted = [], []
    with h5py.File(root / row['trajectory_file'], 'r') as data:
        actions = data['actions'][:]
        phases = [v.decode() if isinstance(v, bytes) else str(v) for v in data['phase'][:]]
        n = len(actions)
        if mask.shape != (n,) or len(phases) != n or len(data['observations/proprio']) != n + 1:
            raise ValueError(f'动作和观测时序不一致: {row["relative_path"]}')
        episode_id = f"{row['task_id']}_demo{int(row['demo']):02d}_{row['mode']}"
        for i, probe in enumerate(probes):
            if probe['return_end'] > n:
                raise ValueError(f'试探越过完整轨迹: {episode_id}')
            for kind in ('intent', 'feedback'):
                step = probe['start'] if kind == 'intent' else probe['return_end']
                if step == n or not bool(mask[step]):
                    omitted.append({'episode_id': episode_id, 'probe_index': i, 'kind': kind,
                                    'step': step, 'reason': 'no_future_action' if step == n else 'context_only_action'})
                    continue
                if kind == 'intent':
                    frames = [max(0, step - 12), max(0, step - 8), max(0, step - 4), step]
                    motion = motion_description(probe, meta, actions[probe['start']:probe['end']])
                else:
                    frames = [max(0, probe['start'] - 4), probe['start'],
                              (probe['start'] + probe['end']) // 2, probe['end'], step]
                    motion = PHASE_INTENTS.get(phases[step], 'Continue manipulating the current target based on observations')
                    if meta['task_language'].lower().startswith('push ') and phases[step] in (
                        'recontact', 'grasp_correction', 'fixture_correction', 'expert_suffix'
                    ):
                        motion = 'Realign edge contact and continue pushing the plate; do not lift the plate'
                oracle = oracle_summary(probe)
                oracle['task_success_at_decision'] = bool(data['observations/success'][step])
                if 'contact_both' in data['observations']:
                    oracle['stable_grasp_at_test_start'] = bool(data['observations/contact_both'][probe['start']])
                elif 'finger_sides_start' in probe:
                    oracle['stable_grasp_at_test_start'] = int(probe['finger_sides_start']) == 2
                events.append(Event(
                    event_id=f'{episode_id}_p{i:02d}_{kind}_t{step}',
                    episode_id=episode_id, task_id=row['task_id'], relative_path=row['relative_path'],
                    kind=kind, step=step, probe_index=i, start=probe['start'], end=probe['end'],
                    return_end=probe['return_end'], frame_steps=sorted(set(frames)),
                    next_phase=phases[step], planned_motion=motion,
                    probe_focus=probe_focus(probe),
                    planned_controls=actions[step:min(step + 8, n)].astype(float).tolist(),
                    oracle=oracle, action_loss_enabled=True,
                ))
    events.sort(key=lambda event: (event.step, event.kind))
    return meta, events, omitted


def temporal_sheet(root: Path, row: dict, event: Event) -> tuple[Image.Image, dict]:
    with h5py.File(root / row['trajectory_file'], 'r') as data:
        observations = data['observations']
        states = observations['proprio'][event.frame_steps].astype(float)
        panels = []
        for timestep in event.frame_steps:
            if timestep > event.step:
                raise ValueError('语言标注请求包含未来帧')
            panels.append([Image.fromarray(observations[f'{camera}_rgb'][timestep])
                           for camera in ('agent', 'wrist')])
        if event.kind == 'feedback':
            before = observations['proprio'][event.start].astype(float)
            after = observations['proprio'][event.end].astype(float)
        else:
            before, after = states[0], states[-1]
        summary = {
            'eef_translation_cm': np.round((after[:3] - before[:3]) * 100, 3).tolist(),
            'gripper_qpos_start': np.round(before[6:], 4).tolist(),
            'gripper_qpos_end': np.round(after[6:], 4).tolist(),
        }
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    # 保留每路原始256像素分辨率，只拼接真实帧并附上时间，不修改场景内容。
    sheet = Image.new('RGB', (512, 282 * len(panels) + 28), 'white')
    painter = ImageDraw.Draw(sheet)
    painter.text((8, 4), f'Fixed camera                     Wrist camera | decision t={event.step}', fill='black', font=font)
    for index, (timestep, panel) in enumerate(zip(event.frame_steps, panels, strict=True)):
        y = 28 + index * 282
        label = f't={timestep}'
        if event.kind == 'intent' and timestep == event.step:
            label += '  BEFORE TEST / CURRENT'
        elif event.kind == 'feedback':
            if timestep == event.start:
                label += '  TEST START'
            if timestep == event.end:
                label += '  TEST RESPONSE'
            if timestep == event.step and timestep > event.end:
                label += '  AFTER RETURN / CURRENT'
            elif timestep == event.step:
                label += ' / CURRENT'
        painter.text((8, y + 3), label, fill='black', font=font)
        sheet.paste(panel[0], (0, y + 26))
        sheet.paste(panel[1], (256, y + 26))
    return sheet, summary


def image_uri(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format='JPEG', quality=95, subsampling=0)
    return 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode('ascii')


def write_inventory(root: Path, destination: Path):
    rows = read_selection(root)
    all_events, omitted, protocols = [], [], Counter()
    for row in rows:
        _, events, skipped = episode_events(root, row)
        protocols[row['validation_protocol']] += 1
        all_events.extend([{
            **event.public(), 'event_id': event.event_id, 'episode_id': event.episode_id, 'task_id': event.task_id,
            'relative_path': event.relative_path, 'oracle': event.oracle,
        } for event in events])
        omitted.extend(skipped)
    summary = {'release_root': str(root.resolve()), 'episodes': len(rows),
               'tasks': len({row['task_id'] for row in rows}), 'events': len(all_events),
               'intent_count': sum(event['kind'] == 'intent' for event in all_events),
               'feedback_count': sum(event['kind'] == 'feedback' for event in all_events),
               'omitted_context_only_or_final': len(omitted), 'protocols': dict(protocols)}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'inventory.json').write_text(json.dumps({'summary': summary, 'events': all_events,
                                                          'omitted': omitted}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
