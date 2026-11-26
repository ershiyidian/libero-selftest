import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path

from client import TeacherClient, load_provider, parse_object
from events import image_uri
from PIL import Image
from prompts import judge_prompt


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    client = TeacherClient(load_provider('buoke'), concurrency=6)
    cases = [json.loads(path.read_text()) for path in sorted((args.source / 'episodes').glob('*.json'))]
    records = [event for episode in cases for event in episode['events'] if 'annotation' in event]
    args.output.mkdir(parents=True, exist_ok=True)

    async def review(event):
        image = Image.open(args.source / event['image_path']).convert('RGB')
        raw = await client.request(judge_prompt(event['packet'], event['annotation'], event['oracle']),
                                   [image_uri(image)], max_output_tokens=1800)
        result = {'event_id': event['event_id'], 'text': event['annotation']['text'],
                  'oracle': event['oracle'], 'review': parse_object(raw['text']), 'response': raw}
        (args.output / (event['event_id'] + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ('response', 'oracle')}, ensure_ascii=False), flush=True)
        return result

    try:
        reviewed = await asyncio.gather(*[review(event) for event in records])
    finally:
        await client.close()
    summary = {'count': len(reviewed), 'verdicts': dict(Counter(row['review']['verdict'] for row in reviewed))}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
