import json
import shutil
from pathlib import Path
from gallery_500 import render_episode

root = Path(__file__).resolve().parent / "render_smoke"
source = Path(__file__).resolve().parent / "cpu_smoke_fixed/episode_000000"
target = root / "episodes/episode_000000/record"
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, target)
result = render_episode(str(root), "episode_000000")
print("NATIVE_ROBOT_PREVIEW_OK", json.dumps({k:v for k,v in result.items() if k!="metadata"},ensure_ascii=False))
