import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, AutoTokenizer

from sft_dataset import AnnotatedEpisode, build_model_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.export / 'manifest.json').read_text())
    summary = manifest['summary']
    if summary.get('annotation_language') != 'en':
        raise AssertionError('Export does not contain English-only language labels')
    release = Path(summary['release_root'])
    tokenizer = AutoTokenizer.from_pretrained(summary['tokenizer_path'])
    counts = {'episodes': 0, 'language_events': 0, 'historical_camera_frames': 0,
              'events_requiring_more_than_8_steps': 0}
    selected = None
    started = time.perf_counter()
    for row in manifest['episodes']:
        dataset = AnnotatedEpisode(release, args.export / row['language_file'])
        mask = np.load(args.export / row['language_mask_file'], allow_pickle=False)
        if int(mask.sum()) != len(dataset.events) or len(dataset.events) != row['language_events']:
            raise AssertionError('语言掩码与已导出事件数量不一致')
        counts['episodes'] += 1
        for event in dataset.events:
            sample = dataset.read(event['step'], max_language_tokens=summary['max_language_tokens_including_eos'])
            if not sample['language_supervised'] or not mask[event['step']]:
                raise AssertionError('事件没有进入真实语言监督掩码')
            if sample['actions'].shape != (8, 7) or sample['history'].shape != (8, 7):
                raise AssertionError('动作窗口形状错误')
            valid = min(8, len(dataset.action_mask) - event['step'])
            np.testing.assert_array_equal(sample['action_mask'][:valid],
                                          dataset.action_mask[event['step']:event['step'] + valid])
            if tokenizer.decode(event['token_ids'][:-1]) != event['text']:
                raise AssertionError('导出文本与词元序列不一致')
            if event['token_ids'][-1] != tokenizer.eos_token_id:
                raise AssertionError('语言没有以原生结束词元终止')
            if max(sample['language_evidence_steps']) > event['step']:
                raise AssertionError('存在未来视觉证据')
            counts['language_events'] += 1
            counts['historical_camera_frames'] += len(sample['language_evidence_images'])
            counts['events_requiring_more_than_8_steps'] += int(sample['requires_more_than_last_8_steps'])
            if selected is None and event['kind'] == 'feedback':
                selected = (sample, event)
    if counts['language_events'] != summary['accepted_language_events']:
        raise AssertionError('全量训练事件数不一致')
    if selected is None:
        raise AssertionError('导出没有反馈样本')
    processor = AutoProcessor.from_pretrained(summary['tokenizer_path'], min_pixels=65536, max_pixels=65536)
    sample, event = selected
    inputs = build_model_inputs(processor, sample)
    changed_targets = {**sample, 'language_text': 'This future answer must never enter the current input.',
                       'language_token_ids': [123], 'actions': np.zeros_like(sample['actions']) + 123}
    unchanged_inputs = build_model_inputs(processor, changed_targets)
    if set(inputs) != set(unchanged_inputs) or any(
        not torch.equal(inputs[key], unchanged_inputs[key]) for key in inputs
    ):
        raise AssertionError('改变监督答案竟然改变了模型输入，存在目标泄漏')
    image_count = int(inputs['image_grid_thw'].shape[0])
    if image_count != len(sample['language_evidence_images']):
        raise AssertionError('原生多模态处理器没有接收完整视觉前缀')
    report = {**counts, 'status': 'passed', 'native_processor_event': event['event_id'],
              'native_processor_images': image_count,
              'native_processor_input_shape': list(inputs['input_ids'].shape),
              'label_token_count_including_eos': event['token_count'],
              'elapsed_seconds': time.perf_counter() - started,
              'action_mask_slices_match_source': True,
              'current_language_is_target_only': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print('VERIFIED_EXPORTED_DATASET', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
