import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from transformers import AutoTokenizer

from prompts import ANNOTATION_LANGUAGE, PROMPT_VERSION


ROOT = Path(__file__).resolve().parent


def export(run: Path, destination: Path, model_dir: Path, max_language_tokens: int = 32):
    config = json.loads((run / 'run_config.json').read_text())
    if config['prompt_version'] != PROMPT_VERSION:
        raise ValueError('该试跑不是当前提示词版本；旧实验只供审计，请用当前版本重新生成')
    if config.get('annotation_language') != ANNOTATION_LANGUAGE:
        raise ValueError('只允许导出当前英文标注，不能混入旧版中文标签')
    episode_paths = sorted((run / 'episodes').glob('*.json'))
    if len(episode_paths) != len(config['selection']):
        raise ValueError('本次运行还有未完成的轨迹，暂不导出不完整训练清单')
    release = Path(config['release'])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest, events_flat, lengths, skipped = [], [], [], []
    for path in episode_paths:
        episode = json.loads(path.read_text())
        with h5py.File(release / episode['trajectory_file'], 'r') as data:
            n = len(data['actions'])
        action_mask = np.load(release / episode['loss_mask_file'], allow_pickle=False)
        language_mask = np.zeros(n, dtype=np.uint8)
        accepted, history = [], []
        for event in episode['events']:
            if event['status'] != 'accepted':
                skipped.append({'event_id': event['event_id'], 'reason': event['status']})
                continue
            if 'event_id' in event['packet'] or 'episode_id' in event['packet']:
                raise ValueError('早期试跑把含结果类别的文件编号放入了教师输入，禁止导出；请用当前提示词重新生成')
            step = event['step']
            if not 0 <= step < n or not bool(action_mask[step]):
                raise ValueError(f'语言标签落在不可监督动作上: {event["event_id"]}')
            if language_mask[step]:
                raise ValueError(f'同一步存在两个语言目标: {event["event_id"]}')
            text = event['annotation']['text']
            from run import english_only
            if not english_only(text) or any(not english_only(event['annotation'].get(k))
                                             for k in ('observation', 'next_action')):
                raise ValueError(f'包含中文或非英文语言标签，拒绝导出: {event["event_id"]}')
            token_ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            lengths.append(len(token_ids))
            if len(token_ids) > max_language_tokens:
                skipped.append({'event_id': event['event_id'], 'reason': 'token_budget_exceeded',
                                'actual_tokens': len(token_ids)})
                continue
            if any(prior['step'] >= step for prior in history):
                raise ValueError('历史文本含当前或未来标签')
            row = {'event_id': event['event_id'], 'step': step, 'kind': event['kind'],
                   'text': text, 'token_ids': token_ids, 'token_count': len(token_ids),
                   'history': history[-3:], 'evidence_steps': event['frame_steps'],
                   'writer_model': event['requested_model'], 'confidence': event['annotation']['confidence'],
                   'annotation_version': config['prompt_version'],
                   'reviewer_model': config['reviewer']['model'],
                   'source_event_file': str((run / 'events' / (event['event_id'] + '.json')).resolve())}
            accepted.append(row)
            language_mask[step] = 1
            history.append({'step': step, 'text': text})
            events_flat.append({'episode_id': episode['episode_id'], 'task_id': episode['task_id'],
                                'trajectory_file': episode['trajectory_file'], **row})
        folder = destination / 'episodes' / episode['episode_id']
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'language_loss_mask.npy', language_mask, allow_pickle=False)
        sidecar = {'episode_id': episode['episode_id'], 'task_id': episode['task_id'],
                   'annotation_language': ANNOTATION_LANGUAGE,
                   'mode': episode['mode'], 'demo': episode['demo'],
                   'trajectory_file': episode['trajectory_file'], 'loss_mask_file': episode['loss_mask_file'],
                   'source_validation_protocol': episode['validation_protocol'],
                   'task_language': episode['task_language'], 'action_count': n,
                   'events': accepted}
        (folder / 'language_annotations.json').write_text(json.dumps(sidecar, ensure_ascii=False, indent=2))
        manifest.append({key: sidecar[key] for key in ('episode_id', 'task_id', 'trajectory_file', 'loss_mask_file')}
                        | {'language_file': str((folder / 'language_annotations.json').relative_to(destination)),
                           'language_mask_file': str((folder / 'language_loss_mask.npy').relative_to(destination)),
                           'language_events': len(accepted)})
    summary = {'release_root': str(release), 'run_root': str(run.resolve()),
               'annotation_language': ANNOTATION_LANGUAGE, 'episodes': len(manifest),
               'tasks': len({row['task_id'] for row in manifest}), 'accepted_language_events': len(events_flat),
               'tokenizer_path': str(model_dir), 'max_language_tokens_including_eos': max_language_tokens,
               'tokens_median': float(np.median(lengths)) if lengths else None,
               'tokens_max': max(lengths) if lengths else None,
               'fits_8_tokens_including_eos': sum(n <= 8 for n in lengths),
               'fits_16_tokens_including_eos': sum(n <= 16 for n in lengths),
               'fits_24_tokens_including_eos': sum(n <= 24 for n in lengths),
               'unexported_by_reason': dict(Counter(row['reason'] for row in skipped))}
    (destination / 'manifest.json').write_text(json.dumps({'summary': summary, 'episodes': manifest,
                                                        'unexported': skipped}, ensure_ascii=False, indent=2))
    with (destination / 'text_events.jsonl').open('w') as file:
        for row in events_flat:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('SFT_EXPORT', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--max-language-tokens', type=int, default=32)
    args = parser.parse_args()
    model = args.model_dir
    if model is None:
        assets = ROOT.parent / 'singlepass_vla/results/assets.json'
        model = Path(json.loads(assets.read_text())['model_path'])
    export(args.run, args.output, model, args.max_language_tokens)
