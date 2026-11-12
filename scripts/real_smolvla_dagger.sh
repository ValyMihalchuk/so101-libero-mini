#!/usr/bin/env bash
# Three-task real SO-101 baseline/DAgger launcher, based on Sang's working residual launcher.
set -eo pipefail
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate "${CONDA_ENV:-lerobot}"
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="${TASK_ID:-cube_flip}"
case "$TASK_ID" in
  cube_flip)
    MODEL_NAME=smolvla_tri_cube_flip_100
    STEP=100000
    EXPECTED_SHA=3eb3b19f2e0c44e953a50154d7f8d4cff94bd3939d74e78637314f32f0c5c2d0
    DEFAULT_POLICY="$PROJECT_ROOT/checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model"
    ;;
  green_push_forward)
    MODEL_NAME=smolvla_tri_green_push_100
    STEP=080000
    EXPECTED_SHA=7c98c434897bffc35720f6e70c5b0af1b1ef9d56637dc8e42cfeae481fc28f5f
    DEFAULT_POLICY="$PROJECT_ROOT/checkpoints/residual_green_push/checkpoints/last/pretrained_model"
    ;;
  cylinder_right_of_cube)
    MODEL_NAME=smolvla_tri_cylinder_right_100
    STEP=080000
    EXPECTED_SHA=75a7b94279589d6940cc70565f8c4c1f0e36f641aada595919fdf0c16a01ad3d
    DEFAULT_POLICY="$PROJECT_ROOT/checkpoints/residual_cylinder_right/checkpoints/080000/pretrained_model"
    ;;
  *)
    echo "TASK_ID must be cube_flip, green_push_forward, or cylinder_right_of_cube" >&2
    exit 2
    ;;
esac
PORT="${PORT:-/dev/tty.usbmodem5B610366661}"
ROBOT_ID="${ROBOT_ID:-my_follower_sim}"
FRONT_CAM="${FRONT_CAM:-3}"
WRIST_CAM="${WRIST_CAM:-0}"
FRONT_W="${FRONT_W:-1280}"
FRONT_H="${FRONT_H:-720}"
WRIST_W="${WRIST_W:-320}"
WRIST_H="${WRIST_H:-240}"
FPS="${FPS:-30}"
DURATION="${DURATION:-15}"
MAX_REL="${MAX_REL:-100}"
DEVICE="${DEVICE:-mps}"
RETURN_HOME="${RETURN_HOME:-1}"
HOME_SECONDS="${HOME_SECONDS:-3}"
HOME_ARGS=(--home-seconds "$HOME_SECONDS")
[ "$RETURN_HOME" = 1 ] || HOME_ARGS+=(--no-return-home)
center() { local s=$(( $1 < $2 ? $1 : $2 )); echo "$(( ($2 - s) / 2 )),$(( ($1 - s) / 2 )),$s,$s"; }
FRONT_CROP="${FRONT_CROP:-$(center "$FRONT_W" "$FRONT_H")}"
WRIST_CROP="${WRIST_CROP:-$(center "$WRIST_W" "$WRIST_H")}"
POLICY="${POLICY:-}"
if [ -z "$POLICY" ]; then
  POLICY="$DEFAULT_POLICY"
  ALTERNATE="$PROJECT_ROOT/outputs/train/$MODEL_NAME/checkpoints/$STEP/pretrained_model"
  if [ ! -d "$POLICY" ] && [ -d "$ALTERNATE" ]; then
    POLICY="$ALTERNATE"
  fi
fi
[ -d "$POLICY" ] || { echo "Expected $TASK_ID checkpoint missing: $POLICY; set POLICY explicitly" >&2; exit 1; }
ACTUAL_SHA="$(python - "$POLICY/model.safetensors" <<'PY'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
)"
[ "$ACTUAL_SHA" = "$EXPECTED_SHA" ] || {
  echo "SmolVLA weights SHA256 mismatch for $TASK_ID: $ACTUAL_SHA; expected $EXPECTED_SHA" >&2
  exit 1
}
DEPLOY="$PROJECT_ROOT/outputs/deploy/${MODEL_NAME}_${STEP}_dagger"
DEPLOY="$(python "$PROJECT_ROOT/scripts/deploy_policy.py" "$POLICY" "$DEPLOY" \
  --crop front="$FRONT_CROP" --crop wrist="$WRIST_CROP")"
exec python "$PROJECT_ROOT/scripts/residual/real_robot_dagger.py" \
  --device "$DEVICE" --port "$PORT" --robot-id "$ROBOT_ID" \
  --front-camera "$FRONT_CAM" --wrist-camera "$WRIST_CAM" \
  --front-width "$FRONT_W" --front-height "$FRONT_H" \
  --wrist-width "$WRIST_W" --wrist-height "$WRIST_H" \
  --fps "$FPS" --duration "$DURATION" --max-rel "$MAX_REL" "${HOME_ARGS[@]}" "$@" \
  --task "$TASK_ID" --policy "$DEPLOY" --source-policy "$POLICY"
