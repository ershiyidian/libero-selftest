from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from benchmarks.group_rollout_queue import (
    initialize, acquire, finish, fail, overview, resume_interrupted,
)


def test_transactional_160_rollout_scheduler_with_real_concurrent_claims(tmp_path):
    database = tmp_path / "macro.sqlite3"
    spec = {"policy": "history-pro", "checkpoint": "ckpt42", "sigma": 0.1}
    initialize(database, tasks=list(range(10)), trials_per_task=16, gpu_count=4, spec=spec)
    initial = overview(database)
    assigned = Counter((row["task_id"], row["assigned_gpu"]) for row in initial["rows"])
    assert len(initial["rows"]) == 160
    assert all(assigned[(task, gpu)] == 4 for task in range(10) for gpu in range(4))

    def worker(rank):
        completed = []
        while (job := acquire(database, gpu=rank // 2, worker=rank)) is not None:
            path = tmp_path / f"task{job['task_id']:02d}_trial{job['trial']:02d}.pt"
            path.write_text("real-output-path-contract")
            finish(database, task_id=job["task_id"], trial=job["trial"], worker=rank,
                   reward=(job["trial"] % 2), control_steps=100+job["trial"],
                   result_path=path)
            completed.append((job["task_id"], job["trial"]))
        return completed

    with ThreadPoolExecutor(max_workers=8) as executor:
        groups = list(executor.map(worker, range(8)))
    keys = [key for group in groups for key in group]
    assert len(keys) == len(set(keys)) == 160
    result = overview(database)
    assert result["done"] == 160 and result["ready_to_train"]
    assert all(g["successes"] == 8 for g in result["groups"].values())
    assert all(g["done"] == 16 for g in result["groups"].values())

    with pytest.raises(ValueError, match="版本|配置"):
        initialize(database, tasks=list(range(10)), trials_per_task=16,
                   gpu_count=4, spec={**spec, "checkpoint": "wrong-policy"})


def test_resume_retries_and_owner_guard(tmp_path):
    path = initialize(tmp_path / "restart.sqlite3", tasks=[2], trials_per_task=4,
                      gpu_count=4, spec={"version": "same"})
    first = acquire(path, gpu=0, worker=7)
    with pytest.raises(ValueError):
        finish(path, task_id=first["task_id"], trial=first["trial"], worker=99,
               reward=1, control_steps=20, result_path=tmp_path / "absent.pt")
    fail(path, task_id=first["task_id"], trial=first["trial"], worker=7,
         error="simulate worker disconnection")
    assert overview(path)["done"] == 0
    second = acquire(path, gpu=0, worker=3)
    assert second is not None
    assert resume_interrupted(path) == 1
    assert overview(path)["groups"][2]["running"] == 0


def test_work_stealing_when_gpu_owns_no_more_jobs(tmp_path):
    db = initialize(tmp_path / "steal.sqlite3", tasks=[0], trials_per_task=16,
                    gpu_count=4, spec={"version": "stable"})
    first = [acquire(db, gpu=0, worker=0) for _ in range(4)]
    assert all(job["assigned_gpu"] == 0 for job in first)
    stolen = acquire(db, gpu=0, worker=0)
    assert stolen["assigned_gpu"] == 1


def test_130_tasks_16_trials_on_three_available_gpus(tmp_path):
    database = initialize(
        tmp_path / "libero130.sqlite3", tasks=range(130),
        trials_per_task=16, gpu_count=3,
        spec={"model": "qwen35-mtp", "policy_version": "smoke"},
    )
    state = overview(database)
    assert state["total"] == 2080
    for task in range(130):
        allocated = Counter(row["assigned_gpu"] for row in state["rows"] if row["task_id"] == task)
        assert allocated == {0: 6, 1: 5, 2: 5}
    with pytest.raises(ValueError, match="配置|版本"):
        initialize(
            database, tasks=range(130), trials_per_task=16,
            gpu_count=3, spec={"model": "qwen35-mtp", "policy_version": "other"},
        )


def test_65_tasks_16_trials_four_gpu_assignment_and_work_stealing(tmp_path):
    tasks = list(range(0, 130, 2))
    database = initialize(
        tmp_path / "65_tasks.sqlite3", tasks=tasks,
        trials_per_task=16, gpu_count=4,
        spec={"model": "qwen35-mtp", "policy_version": "same"},
    )
    status = overview(database)
    assert status["total"] == 1040
    allocation = Counter((row["task_id"], row["assigned_gpu"]) for row in status["rows"])
    assert all(allocation[(task, gpu)] == 4 for task in tasks for gpu in range(4))
    # 先领取本卡配额，4个用完之后从其他GPU偷取工作。
    claimed = [acquire(database, gpu=0, worker=0) for _ in range(65 * 4)]
    assert all(job is not None and job["assigned_gpu"] == 0 for job in claimed)
    stolen = acquire(database, gpu=0, worker=0)
    assert stolen is not None and stolen["assigned_gpu"] in (1, 2, 3)
