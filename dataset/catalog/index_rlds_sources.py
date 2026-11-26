import json
import os
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
from tensorflow.core.example.example_pb2 import Example

ROOT = Path("/data/guest/vla_branch_lab/VLA-SelfTest")
OUT = ROOT / "data_generation/all_tasks/sources"


def main():
    source = ROOT / "official_training/assets/libero"
    summary = {}
    output = OUT / "rlds_episode_sources.jsonl"
    with output.open("w", encoding="utf-8") as target:
        for directory in sorted(source.iterdir()):
            if not directory.is_dir():
                continue
            shards = sorted(directory.rglob("*.tfrecord-*"))
            if not shards:
                continue
            language_counts = Counter()
            row_count = 0
            for shard in shards:
                ds = tf.data.TFRecordDataset(str(shard), num_parallel_reads=1)
                options = tf.data.Options()
                options.threading.private_threadpool_size = 1
                options.threading.max_intra_op_parallelism = 1
                ds = ds.with_options(options)
                for ordinal, raw in enumerate(ds):
                    example = Example()
                    example.ParseFromString(bytes(raw.numpy()))
                    fields = example.features.feature
                    instruction_keys = [key for key in fields if key.endswith("language_instruction")]
                    assert len(instruction_keys) == 1, list(fields)
                    texts = [
                        item.decode("utf-8").strip()
                        for item in fields[instruction_keys[0]].bytes_list.value
                    ]
                    unique = sorted(set(text for text in texts if text))
                    paths = {
                        key: [b.decode("utf-8") for b in value.bytes_list.value]
                        for key, value in fields.items()
                        if "episode_metadata" in key
                    }
                    assert unique
                    row = {
                        "suite": directory.name, "shard": str(shard),
                        "record_index": ordinal, "languages": unique,
                        "steps_from_language": len(texts),
                        "original_episode_metadata": paths,
                        "reuse_roles": ["ordinary_demonstration", "task_prefix", "phase_reference"],
                    }
                    language_counts.update(unique)
                    row_count += 1
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary[directory.name] = {
                "episodes": row_count, "languages": dict(language_counts), "shards": len(shards),
            }
            print("RLDS_PARSED", directory.name, row_count, json.dumps(dict(language_counts)), flush=True)
    (OUT / "rlds_language_counts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
