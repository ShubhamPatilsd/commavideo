#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
EPOCHS=300
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)           IN_DIR="${2%/}";         shift 2 ;;
    --video-names-file|--video_names_file) VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --epochs)                    EPOCHS="$2";              shift 2 ;;
    --device)                    DEVICE="$2";              shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$PD"
python -m submissions.nerv.compress \
  --in-dir "$IN_DIR" \
  --video-names-file "$VIDEO_NAMES_FILE" \
  --epochs "$EPOCHS" \
  --device "$DEVICE"

cd "${HERE}/archive"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
