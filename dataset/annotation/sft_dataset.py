import json
from bisect import bisect_left
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from prompts import ANNOTATION_LANGUAGE
from run import english_only


def build_model_inputs(processor, sample: dict, device='cpu') -> dict:
    import torch

    images = sample['language_evidence_images'] if sample['language_supervised'] else sample['images']
    if sample['language_supervised'] and not images:
        raise ValueError('语言监督样本必须读入原判断对应的历史视觉证据')
    steps = sample['language_evidence_steps'] if sample['language_supervised'] else [
        max(0, sample['step'] - 8), sample['step']]
    content = []
    for index, image in enumerate(images):
        content.append({'type': 'text', 'text': f"t={steps[index // 2]}, {'fixed camera' if index % 2 == 0 else 'wrist camera'}:"})
        content.append({'type': 'image', 'image': image})
    movement = sample['displacement_cm']
    content.append({'type': 'text', 'text': (
        f"Task: {sample['instruction']}\nPrevious thoughts: {sample['previous_text']}\n"
        f"Robot displacement over the last 8 executed actions (cm): "
        f"x={movement[0]:.2f}, y={movement[1]:.2f}, z={movement[2]:.2f}.\n"
        'Use only observations already made. Give a short natural English thought '
        'about the current physical evidence and the next intended action.'
    )})
    messages = [
        {'role': 'system', 'content': 'You are a robot controller. Use natural concise English. '
                                       'Base decisions on observed evidence; say when an effect is uncertain.'},
        {'role': 'user', 'content': content},
    ]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    # 当前语言、未来动作、离线试探成功标签都没有写入渲染的输入提示。
    batch = processor(text=[rendered], images=images, return_tensors='pt')
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


class AnnotatedEpisode:
    def __init__(self, release: Path, language_file: Path):
        self.release = Path(release)
        self.annotation = json.loads(Path(language_file).read_text())
        if self.annotation.get('annotation_language') != ANNOTATION_LANGUAGE:
            raise ValueError('This annotation is not from the current English-only pipeline')
        self.events = self.annotation['events']
        if any(not english_only(event['text']) for event in self.events):
            raise ValueError('English-only annotation contains a non-English language label')
        self.steps = [event['step'] for event in self.events]
        if self.steps != sorted(set(self.steps)):
            raise ValueError('语言事件时间必须严格递增')
        self.action_mask = np.load(self.release / self.annotation['loss_mask_file'], allow_pickle=False)

    def read(self, step: int, max_language_tokens: int | None = None,
             include_language_evidence: bool = True) -> dict:
        index = bisect_left(self.steps, step)
        current = self.events[index] if index < len(self.steps) and self.steps[index] == step else None
        evidence_steps = list(current.get('evidence_steps', [step])) if current else []
        if any(t < 0 or t > step for t in evidence_steps):
            raise ValueError('语言证据必须来自当前或更早的实际观察')
        language_images = []
        with h5py.File(self.release / self.annotation['trajectory_file'], 'r') as file:
            n = len(file['actions'])
            if not 0 <= step < n:
                raise IndexError(step)
            old = max(0, step - 8)
            images = [Image.fromarray(file[f'observations/{camera}_rgb'][t])
                      for t in (old, step) for camera in ('agent', 'wrist')]
            state = file['observations/proprio'][step].astype(np.float32)
            past_state = file['observations/proprio'][old].astype(np.float32)
            history = np.zeros((8, 7), dtype=np.float32)
            count = step - old
            if count:
                history[-count:] = file['actions'][old:step]
            target = np.zeros((8, 7), dtype=np.float32)
            valid = min(n - step, 8)
            target[:valid] = file['actions'][step:step + valid]
            mask = np.zeros(8, dtype=np.float32)
            mask[:valid] = self.action_mask[step:step + valid]
            if include_language_evidence:
                # 退回动作可能让最后8步不含试探响应，语言训练保留实际使用的完整历史片段。
                language_images = [Image.fromarray(file[f'observations/{camera}_rgb'][t])
                                   for t in evidence_steps for camera in ('agent', 'wrist')]
        previous = self.events[max(0, index - 3):index]
        if any(event['step'] >= step for event in previous):
            raise ValueError('训练输入泄漏当前/未来语言标签')
        ids = list(current['token_ids']) if current else []
        if max_language_tokens is not None and len(ids) > max_language_tokens:
            raise ValueError(f'完整句子需要{len(ids)}个词元，预算{max_language_tokens}不足；禁止截断成残句')
        return {'instruction': self.annotation['task_language'], 'step': step, 'images': images,
                'proprio': state, 'history': history, 'actions': target, 'action_mask': mask,
                'displacement_cm': (state[:3] - past_state[:3]) * 100,
                'previous_text': '\n'.join(event['text'] for event in previous) or 'Ready to begin.',
                'language_text': current['text'] if current else None,
                'language_token_ids': ids, 'language_supervised': current is not None,
                'language_evidence_steps': evidence_steps,
                'language_evidence_images': language_images,
                'requires_more_than_last_8_steps': any(t < max(0, step - 8) for t in evidence_steps)}
