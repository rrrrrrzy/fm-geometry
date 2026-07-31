#!/usr/bin/env bash
# Run the GPU capture stages on ONE rollout shard of a recording. These materialize what the
# learned baselines need but the recorder does not store: the K-candidate resample posterior
# (the ground truth), observation embeddings, action-expert hidden features, and the
# Diff-DAgger flow loss. `accel` and `Straightness` need NONE of this — skip it if you only
# want the two free scores.
#
#   fd_gpu_stages.sh <PYBIN> <RUN_PATH> <SHARD_ID> <NUM_SHARDS> <FIRST_ACTIONS> [MICRO_BATCH]
#
# PYBIN is the interpreter to use (e.g. `python`); RUN_PATH is a full path to outputs/runs/<run>.
#
# Shard i owns rollouts {i, i+N, ...}. Every stage writes per-rollout npz and resume-skips
# ones already on disk, so shards are race-free AND a reboot only re-does the in-flight rollout.
# The env (HF cache, offline flags, OMP caps) is set by the caller.
set -uo pipefail

PYBIN="$1"; RUN="$2"; SHARD="$3"; NSHARDS="$4"; FIRST_ACTIONS="$5"; MICRO="${6:-}"
MICRO_ARG=""; [[ -n "$MICRO" ]] && MICRO_ARG="--micro-batch $MICRO"
REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"
RUNDIR="$RUN"                                   # a full path (outputs/runs/<run>)
[[ -d "$RUNDIR/fm" ]] || { echo "FATAL: $RUNDIR/fm not found"; exit 2; }

NROLL=$("$PYBIN" -c "import json; print(len(json.load(open('$RUNDIR/fm/manifest.json'))['rollouts']))")
if ! [[ "$NROLL" =~ ^[0-9]+$ ]]; then echo "FATAL: could not read rollout count ($NROLL)"; exit 2; fi
BATCH_SIZES=$("$PYBIN" -c "import json; m=json.load(open('$RUNDIR/fm/manifest.json')); print(' '.join(str(int(r.get('batch_size', 1))) for r in m['rollouts']))")
read -ra BATCH_SIZE <<< "$BATCH_SIZES"

ALL_IDS=$(seq "$SHARD" "$NSHARDS" $((NROLL-1)))
echo "[shard $SHARD/$NSHARDS] run=$RUNDIR  owns $(echo $ALL_IDS|wc -w) rollouts  fa=$FIRST_ACTIONS  micro=${MICRO:-none}"

pending () {  # <subdir> <prefix> -> ids whose <subdir>/<prefix>ro<id>.npz is absent
  local sub="$1" pre="$2" out=""
  for i in $ALL_IDS; do
    local b="${BATCH_SIZE[$i]:-1}" complete=1
    if [[ "$b" -eq 1 ]]; then
      [[ -f "$RUNDIR/$sub/${pre}ro${i}.npz" ]] || complete=0
    else
      for e in $(seq 0 $((b-1))); do
        [[ -f "$RUNDIR/$sub/${pre}ro${i}_e${e}.npz" ]] || { complete=0; break; }
      done
    fi
    [[ "$complete" -eq 1 ]] || out="$out $i"
  done
  echo "$out"
}

rc=0
run_stage () {  # <label> <subdir> <prefix> <cmd...>
  local label="$1" sub="$2" pre="$3"; shift 3
  local todo; todo=$(pending "$sub" "$pre")
  local n; n=$(echo $todo | wc -w)
  if [[ "$n" -eq 0 ]]; then echo "=== $label: all present, SKIP ==="; return; fi
  echo "=== $label: $n rollouts pending ==="
  "$@" --rollouts $todo --no-progress || { echo "$label FAILED (rc=$?)"; rc=1; }
}

run_stage "[1/4] chunk_divergence (k=32)" chunk_divergence chunk_divergence_ \
    "$PYBIN" cli/divergence.py --run "$RUNDIR" --samples 32 --first-actions "$FIRST_ACTIONS" $MICRO_ARG
run_stage "[2/4] obs_emb" obs_emb obs_emb_ \
    "$PYBIN" cli/obs_emb.py --run "$RUNDIR"
run_stage "[3/4] hidden_states (SAFE)" hidden_states hidden_ \
    "$PYBIN" cli/hidden_states.py --run "$RUNDIR" --horizon-reduce mean --diff-reduce mean
run_stage "[4/4] fm_loss_score" fm_loss fm_loss_ \
    "$PYBIN" cli/fm_loss.py --run "$RUNDIR" --m-t 16 --m-noise 2

echo "[shard $SHARD/$NSHARDS] done rc=$rc"
exit $rc
