import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "runs/selftest500_01"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    pids = [int((RUN / name).read_text()) for name in ("producer.pid", "publisher.pid")]
    (RUN / "STOP").write_text("增加CPU并行数；当前回合正常完成后恢复。", encoding="utf-8")
    print("WAITING_FOR_CURRENT_EPISODES", pids, flush=True)
    start = time.monotonic()
    while any(alive(pid) for pid in pids):
        if time.monotonic() - start > 900:
            raise TimeoutError("现有回合尚未结束，未启动重复生产器")
        time.sleep(5)
    (RUN / "STOP").unlink()
    environment = os.environ.copy()
    environment.update({
        "MUJOCO_GL": "glx", "PYOPENGL_PLATFORM": "glx",
        "MUJOCO_EGL_DEVICE_ID": "0", "LIBGL_ALWAYS_SOFTWARE": "1",
        "GALLIUM_DRIVER": "llvmpipe", "LP_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    with (HERE / "logs/supervisor_resumed.log").open("a") as log:
        child = subprocess.Popen(
            ["nice", "-n", "10", "xvfb-run", "-a", "bash", str(HERE / "resume_jobs.sh")],
            env=environment, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    (RUN / "resumed_supervisor.pid").write_text(str(child.pid))
    print("RESUMED_WITH_8_CPU_WORKERS", child.pid, flush=True)


if __name__ == "__main__":
    main()
