import argparse
import asyncio
import html
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from client import TeacherClient, load_provider, parse_object
from events import DEFAULT_RELEASE, episode_events, image_uri, read_selection, temporal_sheet
from prompts import ANNOTATION_LANGUAGE, PROMPT_VERSION, WRITER_RULES, SEMANTIC_RULES, semantic_prompt, writer_prompt


DEFAULT_TASKS = ['L001', 'L026', 'L031', 'L034', 'L095', 'L125']


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.incomplete')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def language_issues(annotation: dict, event) -> list[str]:
    issues = []
    if annotation.get('status') not in ('ok', 'skip'):
        return ['status must be ok or skip']
    if annotation['status'] == 'skip':
        return [] if annotation.get('skip_reason') and english_only(annotation['skip_reason']) else ['A skip requires an English reason']
    text = annotation.get('text', '')
    if not isinstance(text, str):
        return ['text must be a complete English sentence']
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not 5 <= len(words) <= 32 or len(text) > 220:
        issues.append('Use a short but complete English thought, usually 8 to 22 words')
    if not english_only(text) or re.search(r'[`<>\r\n]', text):
        issues.append('The text must be English only, without Chinese, markup, or newlines')
    if re.search(r'perfectly|flawlessly|definitely|100%|maximum force|stronger force|in the demonstration|expert trajectory|next stage|dataset|annotation', text, flags=re.I):
        issues.append('Avoid theatrical claims, hidden force controls, or dataset narration')
    evidence = annotation.get('evidence_steps')
    if not isinstance(evidence, list) or not evidence or any(type(x) is not int or x not in event.frame_steps for x in evidence):
        issues.append('evidence_steps must refer to the provided images')
    if isinstance(evidence, list) and any(isinstance(x, int) and x > event.step for x in evidence):
        issues.append('A referenced image is later than the current observation')
    for field in ('observation', 'next_action'):
        if not isinstance(annotation.get(field), str) or not english_only(annotation[field]):
            issues.append(field + ' must be a nonempty English explanation')
    if annotation.get('confidence') not in ('clear', 'uncertain'):
        issues.append('confidence must be clear or uncertain')
    return issues


def english_only(text: object) -> bool:
    return (
        isinstance(text, str) and bool(re.search('[A-Za-z]', text))
        and re.search('[\u3400-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7AF]', text) is None
    )


def build_packet(meta: dict, event, motion_summary: dict, history: list[dict]) -> dict:
    return {
        'task': meta['task_language'], **event.public(),
        'kind_explanation': ('The small test has not been executed; state an intention' if event.kind == 'intent'
                             else 'The test and any return motion have occurred; decide from current evidence'),
        'context_history': history[-3:],
        'observed_robot_motion': motion_summary,
        'motion_measurement_interval': [event.frame_steps[0], event.step] if event.kind == 'intent' else [event.start, event.end],
        'test_response_compare_steps': [event.start, event.end] if event.kind == 'feedback' else None,
        'return_motion_after_test': event.kind == 'feedback' and event.return_end > event.end,
        'next_demo_action_intent': event.planned_motion,
        'next_demo_action_note': 'This is the intended future action, not observed future success.',
    }


def validate_semantics(review: dict, event, annotation: dict) -> dict:
    claim = review.get('claim')
    if claim not in ('plan_only', 'positive', 'negative', 'uncertain', 'none'):
        raise ValueError('文字审查必须返回合法claim类型')
    for key in ('slip_or_drop_claim', 'hidden_cause_claim', 'entire_task_complete_claim',
                'probe_subject_match', 'action_match', 'natural'):
        if type(review.get(key)) is not bool:
            raise ValueError(f'语义审查的{key}必须是JSON布尔值')
    issues = list(review.get('issues', []))
    if event.kind == 'feedback':
        if (claim == 'positive' and event.oracle['outcome'] == 'negative') or (
            claim == 'negative' and event.oracle['outcome'] == 'pass'
        ):
            issues.append('Claim contradicts the physical test response; compare TEST START and TEST RESPONSE, not AFTER RETURN. Express genuine uncertainty when hidden.')
        if claim in ('plan_only', 'none'):
            issues.append('Feedback must describe the observed test response or an explicit visual uncertainty.')
    elif claim != 'plan_only':
        issues.append('Before the test, propose a test only; do not claim its future result.')
    if review.get('slip_or_drop_claim') and not event.oracle.get('stable_grasp_at_test_start', False):
        issues.append('There was no stable grasp before the test; do not invent a slip or drop.')
    if review.get('hidden_cause_claim'):
        issues.append('Do not assert hidden friction, weight, material, or grip force.')
    if review.get('entire_task_complete_claim') and not event.oracle.get('task_success_at_decision', False):
        issues.append('The overall task is not complete at this point.')
    if review.get('action_match') is not True:
        issues.append('The proposed action does not match the subsequent action.')
    if review.get('natural') is not True:
        issues.append('Use natural conversational English, without annotation jargon or exaggeration.')
    if review.get('probe_subject_match') is not True:
        issues.append('The feedback confuses the current tested object with another task object.')
    return {'verdict': 'rewrite' if issues else 'accept', 'issues': issues,
            'grounding': 'uncertain' if claim == 'uncertain' else ('contradiction' if issues else 'physics_consistent'),
            'claim': claim, 'reason': '; '.join(issues) if issues else '文字语义与当前物理反馈及动作计划相符'}


def select_rows(rows: list[dict], tasks: list[str] | None, per_mode: int, limit: int | None):
    if per_mode < 1 or (limit is not None and limit < 1):
        raise ValueError('每种类型采样数和轨迹数量上限必须为正')
    selected = []
    counts = Counter()
    tasks = set(tasks) if tasks else None
    for row in rows:
        if tasks and row['task_id'] not in tasks:
            continue
        key = (row['task_id'], row['mode'])
        if counts[key] >= per_mode:
            continue
        counts[key] += 1
        selected.append(row)
    if tasks:
        missing = tasks - {row['task_id'] for row in selected}
        if missing:
            raise ValueError(f'选择的任务没有轨迹: {sorted(missing)}')
    return selected[:limit] if limit else selected


async def annotate_episode(root: Path, row: dict, destination: Path, client: TeacherClient,
                           reviewer: TeacherClient | None, max_rewrites: int, resume: bool):
    meta, events, omitted = episode_events(root, row)
    episode_id = events[0].episode_id if events else f"{row['task_id']}_{row['demo']}"
    target = destination / 'episodes' / f'{episode_id}.json'
    if target.exists() and resume:
        saved = json.loads(target.read_text())
        if saved.get('annotation_language') == ANNOTATION_LANGUAGE and all(
            x['status'] in ('accepted', 'skipped') and (
                x['status'] != 'accepted' or english_only(x.get('annotation', {}).get('text'))
            ) for x in saved['events']
        ):
            print('CACHED', episode_id, flush=True)
            return saved
    records, history = [], []
    for event in events:
        started = time.perf_counter()
        image, summary = temporal_sheet(root, row, event)
        image_path = destination / 'evidence' / f'{event.event_id}.jpg'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path, quality=95, subsampling=0)
        packet = build_packet(meta, event, summary, history)
        cached_event = destination / 'events' / (event.event_id + '.json')
        if resume and cached_event.exists():
            saved_event = json.loads(cached_event.read_text())
            if saved_event.get('packet') == packet and saved_event.get('status') in ('accepted', 'skipped') and (
                saved_event['status'] != 'accepted' or english_only(saved_event.get('annotation', {}).get('text'))
            ):
                records.append(saved_event)
                if saved_event['status'] in ('accepted', 'draft'):
                    history.append({'step': event.step, 'text': saved_event['annotation']['text']})
                print('CACHED_EVENT', event.event_id, flush=True)
                continue
        record = {**asdict(event), 'provider': client.provider.name,
                  'requested_model': client.provider.model,
                  'packet': packet, 'image_path': str(image_path.relative_to(destination)),
                  'attempts': [], 'status': 'error'}
        correction = None
        for index in range(max_rewrites + 1):
            attempt = {'index': index + 1}
            try:
                response = await client.request(writer_prompt(packet, correction=correction), [image_uri(image)])
                attempt['writer'] = response
                annotation = parse_object(response['text'])
                attempt['annotation'] = annotation
                issues = language_issues(annotation, event)
                if issues:
                    attempt['program_issues'] = issues
                    correction = '; '.join(issues)
                elif annotation['status'] == 'skip':
                    record['status'] = 'skipped'
                    record['skip_reason'] = annotation['skip_reason']
                else:
                    if reviewer:
                        review_response = await reviewer.request(semantic_prompt(packet, annotation),
                                                                 max_output_tokens=2048)
                        attempt['reviewer'] = review_response
                        review = parse_object(review_response['text'])
                        attempt['semantic_extraction'] = review
                        review = validate_semantics(review, event, annotation)
                        if review.get('verdict') not in ('accept', 'rewrite', 'skip'):
                            raise ValueError('审查结果字段不合法')
                        attempt['review'] = review
                    else:
                        review = {'verdict': 'accept', 'reason': '未运行独立审查，需人工复核'}
                    if review['verdict'] == 'accept':
                        record['status'] = 'accepted' if reviewer else 'draft'
                        record['annotation'] = annotation
                    elif review['verdict'] == 'skip':
                        record['status'] = 'skipped'
                        record['skip_reason'] = review.get('reason', '')
                    else:
                        correction = '; '.join(review.get('issues', [])) or review.get('reason', '')
                record['attempts'].append(attempt)
                if record['status'] in ('accepted', 'draft', 'skipped'):
                    break
                record['status'] = 'rejected'
            except Exception as error:
                attempt['error_type'] = type(error).__name__
                attempt['error'] = str(error)
                record['attempts'].append(attempt)
                record['status'] = 'error'
                # 模型格式错误可以局部再试；认证或服务故障留在失败清单，不伪造标签。
                if not isinstance(error, (ValueError, json.JSONDecodeError)):
                    break
                correction = 'Return only the specified JSON object with English strings; no Markdown.'
        record['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(destination / 'events' / (event.event_id + '.json'), record)
        if record['status'] in ('accepted', 'draft'):
            history.append({'step': event.step, 'text': record['annotation']['text']})
        records.append(record)
        print('EVENT', json.dumps({'event_id': event.event_id, 'provider': client.provider.name,
                                   'status': record['status'], 'text': record.get('annotation', {}).get('text'),
                                   'attempts': len(record['attempts'])}, ensure_ascii=False), flush=True)
    result = {'episode_id': episode_id, 'task_id': row['task_id'], 'demo': row['demo'],
              'annotation_language': ANNOTATION_LANGUAGE,
              'mode': row['mode'], 'task_language': meta['task_language'],
              'release_relative_path': row['relative_path'], 'validation_protocol': row['validation_protocol'],
              'trajectory_file': row['trajectory_file'], 'loss_mask_file': row['loss_mask_file'],
              'events': records, 'omitted': omitted}
    atomic_json(target, result)
    return result


def summarize(results: list[dict], destination: Path, elapsed: float):
    events = [event for episode in results for event in episode['events']]
    states = Counter(event['status'] for event in events)
    accepted = [event for event in events if event['status'] in ('accepted', 'draft')]
    texts = [event['annotation']['text'] for event in accepted]
    costs, latencies = Counter(), []
    for event in events:
        for attempt in event['attempts']:
            for role in ('writer', 'reviewer'):
                response = attempt.get(role)
                if response:
                    usage = response.get('usage') or {}
                    costs[(response['provider'], 'calls')] += 1
                    for key in ('input_tokens', 'output_tokens'):
                        costs[(response['provider'], key)] += usage.get(key, 0)
                    latencies.append(response['elapsed_seconds'])
    summary = {'annotation_language': ANNOTATION_LANGUAGE,
               'episodes': len(results), 'tasks': len({row['task_id'] for row in results}),
               'events': len(events), 'statuses': dict(states), 'unique_texts': len(set(texts)),
               'rewritten_events': sum(len(event['attempts']) > 1 for event in events),
               'wall_seconds': elapsed, 'usage': {f'{k[0]}.{k[1]}': v for k, v in costs.items()},
               'mean_request_seconds': sum(latencies) / len(latencies) if latencies else 0,
               'failure_events': [event['event_id'] for event in events if event['status'] in ('error', 'rejected')],
               'skipped_events': [event['event_id'] for event in events if event['status'] == 'skipped']}
    previous = destination / 'summary.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        summary['wall_seconds'] += old.get('wall_seconds', 0)
    summary['last_invocation_seconds'] = elapsed
    atomic_json(destination / 'summary.json', summary)
    gallery(results, summary, destination)
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def gallery(results: list[dict], summary: dict, destination: Path):
    # Large runs may contain >2800 events; split preview by task instead of
    # placing every raw API response into one giant browser DOM.
    sections_by_task = {}
    escape = html.escape
    for result in results:
        cards = []
        for event in result['events']:
            annotation = event.get('annotation', {})
            details = {'context_history': event['packet']['context_history'],
                       'frame_steps': event['frame_steps'], 'oracle': event['oracle'],
                       'attempts': event['attempts']}
            cards.append(f'''<article class="event {escape(event['status'])}">
<div class="eventhead"><span>t={event['step']} · {escape(event['kind'])}</span><b>{escape(event['status'])}</b></div>
<h3>{escape(annotation.get('text', event.get('skip_reason', '未通过')))}</h3>
<p>{escape(annotation.get('observation', ''))}</p>
<details><summary>查看截至当前时刻的双视角证据</summary><img loading="lazy" src="../{escape(event['image_path'])}"></details>
<details><summary>原始生成、审查和历史</summary><pre>{escape(json.dumps(details, ensure_ascii=False, indent=2))}</pre></details>
</article>''')
        sections_by_task.setdefault(result['task_id'], []).append(f'''<section><h2>{escape(result['task_id'])} · demo {result['demo']} · {escape(result['mode'])}</h2>
<p class="task">{escape(result['task_language'])}</p><div class="events">{''.join(cards)}</div></section>''')
    header = '''<!DOCTYPE html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-Test English Annotations</title><style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:30px auto;padding:0 20px;background:#f5f6f8;color:#19202c}}
h1{{font-size:28px}}h2{{font-size:20px}}h3{{font-size:20px;line-height:1.6}}section{{margin:32px 0}}
.task{{color:#526071}}.events{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}}
.event{{background:white;border:1px solid #dce1e9;border-radius:12px;padding:18px}}.eventhead{{display:flex;justify-content:space-between;color:#566579;font-size:13px}}
.accepted{{border-top:4px solid #28835b}}.rejected,.error{{border-top:4px solid #b34c48}}.draft{{border-top:4px solid #9c8040}}
img{{max-width:100%;height:auto}}pre{{font-size:12px;max-height:400px;overflow:auto;white-space:pre-wrap}}summary{{cursor:pointer;margin:12px 0}}
</style><h1>Self-Test English Annotations</h1>'''.replace('{{','{').replace('}}','}')
    (destination / 'preview').mkdir(exist_ok=True)
    nav = []
    for task_id, sections in sorted(sections_by_task.items()):
        page = f'{task_id}.html'
        (destination / 'preview' / page).write_text(
            header + '<p><a href="index.html">All tasks</a></p>'
            + ''.join(sections) + '</html>'
        )
        nav.append(f'<li><a href="{escape(page)}">{escape(task_id)}</a> ({len(sections)} episodes)</li>')
    (destination / 'preview/index.html').write_text(
        header + f"<p>{summary['tasks']} tasks · {summary['episodes']} episodes · {summary['events']} decisions.</p>"
        + '<p>Evidence images never extend beyond the decision time.</p>'
        + '<ul>' + ''.join(nav) + '</ul></html>'
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--provider', choices=('buoke', 'ldc'), default='buoke')
    parser.add_argument('--reviewer', choices=('buoke', 'ldc', 'none'), default='buoke')
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    parser.add_argument('--all-tasks', action='store_true')
    parser.add_argument('--per-mode', type=int, default=1)
    parser.add_argument('--limit-episodes', type=int)
    parser.add_argument('--max-rewrites', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--episode-workers', type=int, default=6,
                        help='Bound concurrent HDF5 readers and episode annotation tasks.')
    args = parser.parse_args()
    if args.max_rewrites < 0:
        raise ValueError('重写次数不能小于0')
    if not 1 <= args.episode_workers <= args.concurrency:
        raise ValueError('episode-workers must be between 1 and provider concurrency')
    rows = select_rows(read_selection(args.release), None if args.all_tasks else args.tasks,
                       args.per_mode, args.limit_episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    provider = load_provider(args.provider)
    client = TeacherClient(provider, args.concurrency)
    reviewer = None
    if args.reviewer == args.provider:
        reviewer = client
    elif args.reviewer != 'none':
        reviewer = TeacherClient(load_provider(args.reviewer), min(args.concurrency, load_provider(args.reviewer).concurrency))
    config = {'release': str(args.release.resolve()), 'writer': provider.public(),
              'annotation_language': ANNOTATION_LANGUAGE,
              'reviewer': reviewer.provider.public() if reviewer else None, 'prompt_version': PROMPT_VERSION,
              'writer_rules': WRITER_RULES, 'semantic_rules': SEMANTIC_RULES,
              'max_rewrites': args.max_rewrites, 'selection': [row['relative_path'] for row in rows]}
    config_path = args.output / 'run_config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config or not args.resume:
            raise ValueError('输出目录已有运行：恢复要求模型、提示词、发布集和清单完全一致')
    else:
        atomic_json(config_path, config)
    began = time.perf_counter()
    # 每条轨迹内部按时间顺序生成；并发只发生在不同轨迹之间。
    episode_semaphore = asyncio.Semaphore(args.episode_workers)
    progress = Counter()
    progress_path = args.output / 'progress.json'

    async def bounded_episode(row):
        async with episode_semaphore:
            result = await annotate_episode(args.release, row, args.output, client, reviewer,
                                            args.max_rewrites, args.resume)
            progress['completed_episodes'] += 1
            progress['completed_events'] += len(result['events'])
            progress.update(Counter(event['status'] for event in result['events']))
            if progress['completed_episodes'] % 10 == 0 or progress['completed_episodes'] == len(rows):
                snapshot = {'language': ANNOTATION_LANGUAGE,
                            'selected_episodes': len(rows),
                            'completed_episodes': progress['completed_episodes'],
                            'completed_events': progress['completed_events'],
                            'status_counts': {k:v for k,v in progress.items()
                                              if k not in ('completed_episodes','completed_events')},
                            'elapsed_seconds': round(time.perf_counter() - began, 2)}
                atomic_json(progress_path, snapshot)
                print('PROGRESS', json.dumps(snapshot, ensure_ascii=False), flush=True)
            return result

    try:
        results = await asyncio.gather(*[
            bounded_episode(row)
            for row in rows
        ])
    finally:
        await client.close()
        if reviewer and reviewer is not client:
            await reviewer.close()
    summarize(results, args.output, time.perf_counter() - began)


if __name__ == '__main__':
    asyncio.run(main())
