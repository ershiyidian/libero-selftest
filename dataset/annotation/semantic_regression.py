import asyncio
import json
from pathlib import Path

from client import TeacherClient, load_provider, parse_object
from prompts import semantic_prompt


CASES = [
    ("The knob seems to have turned a little, but I can't tell if it's fully on.", 'positive'),
    ("I can't tell whether the knob turned. I'll adjust my position.", 'uncertain'),
    ("I can't see whether the drawer moved. I'll pull gently again.", 'uncertain'),
    ("The door opened a little when I pulled it. I'll keep pulling.", 'positive'),
    ("The black bowl stayed on the table. I'll realign my gripper.", 'negative'),
    ("I'll lift the bowl slightly to see if my grip holds.", 'plan_only'),
]


async def main():
    client = TeacherClient(load_provider('ldc'), concurrency=4)
    destination = Path(__file__).resolve().parent / 'reviews/semantic_regression_final.json'

    async def evaluate(text, expected):
        packet = {'kind': 'intent' if expected == 'plan_only' else 'feedback',
                  'task': 'Test the response of an object to a small physical probe.',
                  'next_demo_action_intent': 'Adjust contact or continue the task.',
                  'planned_control_mean_xyz': [0.0, 0.0, 0.0]}
        attempts = []
        for index in range(2):
            response = await client.request(semantic_prompt(packet, {'text': text}), max_output_tokens=2048)
            attempts.append(response)
            try:
                extracted = parse_object(response['text'])
                return {'text': text, 'expected_claim': expected, 'observed_claim': extracted.get('claim'),
                        'pass': extracted.get('claim') == expected, 'attempts': attempts}
            except ValueError as error:
                response['format_error'] = str(error)
        return {'text': text, 'expected_claim': expected, 'observed_claim': None,
                'pass': False, 'attempts': attempts}

    try:
        rows = await asyncio.gather(*(evaluate(text, label) for text, label in CASES))
    finally:
        await client.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != 'attempts'}, ensure_ascii=False), flush=True)
    if not all(row['pass'] for row in rows):
        raise AssertionError('真实文本语义回归失败，见保存的模型响应')


if __name__ == '__main__':
    asyncio.run(main())
