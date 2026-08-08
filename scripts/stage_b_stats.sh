#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

link() {
  ln -s "$ROOT/$1" "$TMP/$2"
}

# Live-A (current HEAD): seed1995 uses the diag2 rerun as the canonical run.
link live_a_aggregate_b_inr_seed1995_diag2.log live_inr_seed1995.log
link live_a_aggregate_b_inr_seed1.log live_inr_seed1.log
link live_a_aggregate_b_inr_seed2.log live_inr_seed2.log
link live_a_aggregate_b_inr_seed3.log live_inr_seed3.log
link live_a_aggregate_b_c100_seed1993.log live_c100_seed1993.log
link live_a_aggregate_b_c100_seed1.log live_c100_seed1.log
link live_a_aggregate_b_c100_seed2.log live_c100_seed2.log
link live_a_aggregate_b_c100_seed3.log live_c100_seed3.log

# EXP-009 reruns under current HEAD.
link sa_sdlora_proto_inr_seed1995_paired_rerun.log exp009_inr_seed1995.log
link sa_sdlora_proto_inr_seed1_paired_rerun.log exp009_inr_seed1.log
link sa_sdlora_proto_inr_seed2_paired_rerun.log exp009_inr_seed2.log
link sa_sdlora_proto_inr_seed3_paired_rerun.log exp009_inr_seed3.log
link sa_sdlora_proto_c100_seed1993_paired_rerun.log exp009_c100_seed1993.log
link sa_sdlora_proto_c100_seed1_paired_rerun.log exp009_c100_seed1.log
link sa_sdlora_proto_c100_seed2_paired_rerun.log exp009_c100_seed2.log
link sa_sdlora_proto_c100_seed3_paired_rerun.log exp009_c100_seed3.log

# Original SD-LoRA baselines (rerun2).
link sdlora_inr_seed1995_proto_baseline_paired_rerun.log sdlora_inr_seed1995.log
link sdlora_inr_seed1_proto_baseline_paired_rerun.log sdlora_inr_seed1.log
link sdlora_inr_seed2_proto_baseline_paired_rerun.log sdlora_inr_seed2.log
link sdlora_inr_seed3_proto_baseline_paired_rerun.log sdlora_inr_seed3.log
link sdlora_c100_seed1993_proto_baseline_paired_rerun.log sdlora_c100_seed1993.log
link sdlora_c100_seed1_proto_baseline_paired_rerun.log sdlora_c100_seed1.log
link sdlora_c100_seed2_proto_baseline_paired_rerun.log sdlora_c100_seed2.log
link sdlora_c100_seed3_proto_baseline_paired_rerun.log sdlora_c100_seed3.log

cd "$ROOT"
run_pair() {
  local metric="$1"
  local ds="$2"
  local a="$3"
  local b="$4"
  python scripts/multiseed_stats.py \
    --group "live_${ds}" "$TMP/live_${ds}_seed*.log" \
    --group "exp009_${ds}" "$TMP/exp009_${ds}_seed*.log" \
    --group "sdlora_${ds}" "$TMP/sdlora_${ds}_seed*.log" \
    --metric "$metric" \
    --paired "$a" "$b" --margin 0.5
}

for metric in final avg forgetting; do
  echo "########## INR ${metric}: Live-A vs EXP-009 ##########"
  run_pair "$metric" inr live_inr exp009_inr
  echo
  echo "########## INR ${metric}: Live-A vs SD-LoRA ##########"
  run_pair "$metric" inr live_inr sdlora_inr
  echo
  echo "########## C100 ${metric}: Live-A vs EXP-009 ##########"
  run_pair "$metric" c100 live_c100 exp009_c100
  echo
  echo "########## C100 ${metric}: Live-A vs SD-LoRA ##########"
  run_pair "$metric" c100 live_c100 sdlora_c100
  echo
done
