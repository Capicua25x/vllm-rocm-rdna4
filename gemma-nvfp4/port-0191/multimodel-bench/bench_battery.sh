#!/bin/bash
# Full bench battery for ONE model already serving on :8011 as alias 'qwen'.
# Writes structured results to results/<label>/.
# Usage: bench_battery.sh <label>          e.g. bench_battery.sh qwenbase
#
# Env:
#   PYTHON          python interpreter (default: python3)
#   BENCH_DIR       dir holding throughput_sweep.sh (the throughput sweeper); default: this script's dir
#   REGRESSION_CMD  optional downstream regression command (e.g. an external task-accuracy suite that
#                   drives the model through your API). Unset = the regression leg is skipped. The command
#                   is run via `bash -c`, with a 3600s cap, output captured to results/<label>/regression.log.
#                   Keep any deployment-specific path in this env var, NOT hard-coded here.
set -u
LABEL="$1"
PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="${BENCH_DIR:-$HERE}"
OUT="$HERE/results/$LABEL"; mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] [$LABEL] $*"; }
mark(){ echo "===BATTERY $LABEL $*==="; }

mark START
# fast capability first
log "image...";   BENCH_NONCE="$LABEL" timeout 180 "$PY" "$HERE/fill_compact_image.py" image   > "$OUT/image.json"   2>&1; mark IMAGE_DONE
                  "$PY" "$HERE/fill_compact_image.py" audio  > "$OUT/audio.json"  2>&1; mark AUDIO_DONE
# timed long-context legs (unique nonce per leg -> cold, no cross-contamination)
log "fill (timed max prefill ~250k)...";  BENCH_NONCE="$LABEL" timeout 800 "$PY" "$HERE/fill_compact_image.py" fill 250    > "$OUT/fill.json"    2>&1; mark FILL_DONE
log "compact (timed round-trip ~180k)..."; BENCH_NONCE="$LABEL" timeout 800 "$PY" "$HERE/fill_compact_image.py" compact 180 > "$OUT/compact.json" 2>&1; mark COMPACT_DONE
# throughput (per-user + aggregate tok/s sweep against :8011)
log "regular throughput...";  ( cd "$BENCH_DIR" && timeout 900 ./throughput_sweep.sh --model qwen )                                   > "$OUT/regular.log" 2>&1; mark REGULAR_DONE
log "6k throughput...";       ( cd "$BENCH_DIR" && timeout 900 ./throughput_sweep.sh --model qwen --prompt-tokens 6000 --levels "1 16 32 64" ) > "$OUT/6k.log" 2>&1; mark 6K_DONE
# Optional downstream task-accuracy regression (longest leg). Configured entirely via $REGRESSION_CMD.
# -u (unbuffered) so the log streams and partial results survive a timeout kill; 60min cap.
if [ -n "${REGRESSION_CMD:-}" ]; then
  log "regression (downstream suite via \$REGRESSION_CMD)..."
  timeout 3600 bash -c "$REGRESSION_CMD" > "$OUT/regression.log" 2>&1
  mark REGRESSION_DONE
else
  log "regression skipped (set REGRESSION_CMD to enable)"; mark REGRESSION_SKIPPED
fi
mark COMPLETE
log "battery done -> $OUT"
