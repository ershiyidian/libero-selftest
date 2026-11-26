set -euo pipefail
ROOT="/data/guest/vla_branch_lab/VLA-SelfTest/data_generation/pipeline"
RUN="$ROOT/runs/selftest500_01"
PYTHON="/home/guest/miniconda3/envs/vla-branch/bin/python"
LOGS="$ROOT/scale500/logs"
mkdir -p "$LOGS"
"$PYTHON" -u "$ROOT/scale500/generate_500.py" --output "$RUN" --target 500 --workers 8 --max-attempts 3000 --resume >>"$LOGS/generation.log" 2>&1 &
generator=$!
"$PYTHON" -u "$ROOT/scale500/publish_500.py" --root "$RUN" --workers 2 --producer-pid "$generator" >>"$LOGS/preview.log" 2>&1 &
publisher=$!
echo "GENERATOR_PID=$generator PUBLISHER_PID=$publisher"
trap 'kill "$generator" "$publisher" 2>/dev/null || true' TERM INT
wait "$generator"
wait "$publisher"
echo "SELFTEST_500_GENERATION_AND_PREVIEW_FINISHED"
