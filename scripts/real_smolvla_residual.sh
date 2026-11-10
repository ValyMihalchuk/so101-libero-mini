#!/usr/bin/env bash
# Cube-flip real follower rollout. Extra options: --mode, --strength, --inject-offset,
# --dry-run, --trial-id, --placement-id. Inherits real_smolvla.sh hardware defaults.
# After DURATION runs out the arm moves to the home pose over HOME_SECONDS (RETURN_HOME=0 disables it);
# after an error or Ctrl-C it holds where it is.
set -eo pipefail
eval "$("${CONDA_EXE:-conda}" shell.bash hook)"
conda activate "${CONDA_ENV:-lerobot}"
set -u
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
    POLICY="$PROJECT_ROOT/checkpoints/smolvla_cube_flip/checkpoints/100000/pretrained_model"
    if [ ! -d "$POLICY" ]; then
        POLICY="$PROJECT_ROOT/outputs/train/smolvla_tri_cube_flip_100/checkpoints/100000/pretrained_model"
    fi
fi
[ -d "$POLICY" ] || { echo "cube_flip 100k checkpoint missing: $POLICY" >&2; exit 1; }
DEPLOY="$PROJECT_ROOT/outputs/deploy/smolvla_tri_cube_flip_100_100000_residual"
DEPLOY="$(python "$PROJECT_ROOT/scripts/deploy_policy.py" "$POLICY" "$DEPLOY" \
  --crop front="$FRONT_CROP" --crop wrist="$WRIST_CROP")"
exec python "$PROJECT_ROOT/scripts/residual/real_robot_residual.py" \
  --policy "$DEPLOY" --device "$DEVICE" --port "$PORT" --robot-id "$ROBOT_ID" \
  --front-camera "$FRONT_CAM" --wrist-camera "$WRIST_CAM" \
  --front-width "$FRONT_W" --front-height "$FRONT_H" \
  --wrist-width "$WRIST_W" --wrist-height "$WRIST_H" \
  --fps "$FPS" --duration "$DURATION" --max-rel "$MAX_REL" "${HOME_ARGS[@]}" "$@"
