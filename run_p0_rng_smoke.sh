#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
NPROC="${NPROC:-$(awk -F',' '{print NF}' <<< "$GPU_IDS")}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1
export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
P0_VERIFY_ONLY="${P0_VERIFY_ONLY:-}"

control_r1=exps/live_a_rng_smoke_control_c100_seed1.json
control_r2=exps/live_a_rng_smoke_control_r2_c100_seed1.json
dual=exps/live_a_rng_smoke_dual_b_c100_seed1.json

run_one() {
  local name="$1" config="$2" outdir="$3" logfile="$4"
  local lock="${outdir%/}.lock"
  if [ -e "$outdir" ]; then
    echo "SMOKE FAIL: $name output directory already exists: $outdir"
    exit 1
  fi
  if ! flock -n "$lock" true 2>/dev/null; then
    echo "SMOKE FAIL: $name filepath lock is held: $lock"
    exit 1
  fi
  echo "===== $(date '+%F %T') START $name ====="
  echo "config=$config outdir=$outdir gpus=$GPU_IDS nproc=$NPROC"
  echo "commit=$(git rev-parse HEAD) config_sha=$(sha256sum "$config" | awk '{print $1}')"
  torchrun --standalone --nproc_per_node="$NPROC" \
    main.py --config="./$config" > "$logfile" 2>&1
  local status=$?
  echo "===== $(date '+%F %T') END $name status=$status ====="
  if [ "$status" -ne 0 ]; then
    echo "SMOKE FAIL: $name exited with status $status"
    exit 1
  fi
  if grep -qiE 'non-determin|warn_only=True|memory efficient attention defaults to a non' "$logfile"; then
    echo "SMOKE FAIL: $name log contains a determinism warning"
    grep -niE 'non-determin|warn_only=True|memory efficient attention defaults' "$logfile" | head -20
    exit 1
  fi
}

if [ -z "$P0_VERIFY_ONLY" ]; then
  run_one CONTROL_R1 "$control_r1" "./CF100_P0_CONTROL_CLEAN_R1/" live_a_rng_smoke_control_r1.log
  run_one CONTROL_R2 "$control_r2" "./CF100_P0_CONTROL_CLEAN_R2/" live_a_rng_smoke_control_r2.log
  run_one DUAL_B_R1 "$dual" "./CF100_P0_DUAL_B_CLEAN_R1/" live_a_rng_smoke_dual_b_r1.log
fi

post_train() {
  grep -o '\[PostTrainHash\] task [0-9]* hash=[0-9a-f]* rng=[0-9a-f]*' "$1" | sort
}
post_eval() {
  grep -o '\[PostEvalHash\] task [0-9]* hash=[0-9a-f]* rng=[0-9a-f]*' "$1" | sort
}
eval_rng_hashes() {
  grep -o '\[RNGHash\] task [0-9]* eval before=[0-9a-f]* after=[0-9a-f]* PASS' "$1" | sort
}
all_rng_hashes() {
  grep -o '\[RNGHash\] task [0-9]* [a-z]* before=[0-9a-f]* after=[0-9a-f]* PASS' "$1" | sort
}
metrics() {
  grep '^Average Accuracy (CNN):' "$1" | sed 's/.*: //' | tr '\n' ' '
}

echo "===== R1/R2 trajectory comparison ====="
echo "R1 post_train:"
post_train live_a_rng_smoke_control_r1.log
echo "R2 post_train:"
post_train live_a_rng_smoke_control_r2.log
if [ "$(post_train live_a_rng_smoke_control_r1.log)" != "$(post_train live_a_rng_smoke_control_r2.log)" ]; then
  echo "SMOKE FAIL: CONTROL R1/R2 post_train hashes differ"
  exit 1
fi
echo "R1 post_eval:"
post_eval live_a_rng_smoke_control_r1.log
echo "R2 post_eval:"
post_eval live_a_rng_smoke_control_r2.log
if [ "$(post_eval live_a_rng_smoke_control_r1.log)" != "$(post_eval live_a_rng_smoke_control_r2.log)" ]; then
  echo "SMOKE FAIL: CONTROL R1/R2 post_eval hashes differ"
  exit 1
fi
echo "R1 rng:"
all_rng_hashes live_a_rng_smoke_control_r1.log
echo "R2 rng:"
all_rng_hashes live_a_rng_smoke_control_r2.log
if [ "$(all_rng_hashes live_a_rng_smoke_control_r1.log)" != "$(all_rng_hashes live_a_rng_smoke_control_r2.log)" ]; then
  echo "SMOKE FAIL: CONTROL R1/R2 RNG hashes differ"
  exit 1
fi
m1="$(metrics live_a_rng_smoke_control_r1.log)"
m2="$(metrics live_a_rng_smoke_control_r2.log)"
echo "R1 metrics: $m1"
echo "R2 metrics: $m2"
if [ "$m1" != "$m2" ]; then
  echo "SMOKE FAIL: CONTROL R1/R2 metrics differ"
  exit 1
fi

echo "===== CONTROL vs DUAL-B trajectory comparison ====="
echo "control post_train:"
post_train live_a_rng_smoke_control_r1.log
echo "dual post_train:"
post_train live_a_rng_smoke_dual_b_r1.log
if [ "$(post_train live_a_rng_smoke_control_r1.log)" != "$(post_train live_a_rng_smoke_dual_b_r1.log)" ]; then
  echo "SMOKE FAIL: CONTROL/Dual-B post_train hashes differ"
  exit 1
fi
echo "control rng:"
eval_rng_hashes live_a_rng_smoke_control_r1.log
echo "dual rng:"
eval_rng_hashes live_a_rng_smoke_dual_b_r1.log
if [ "$(eval_rng_hashes live_a_rng_smoke_control_r1.log)" != "$(eval_rng_hashes live_a_rng_smoke_dual_b_r1.log)" ]; then
  echo "SMOKE FAIL: CONTROL/Dual-B RNG hashes differ"
  exit 1
fi

dual_rank_lines=$(grep '\[DualHead\] rank ' live_a_rng_smoke_dual_b_r1.log | sed 's/.*rank /rank /' | sort)
echo "dual rank sync:"
echo "$dual_rank_lines"
rank_task0=$(grep '\[DualHead\] rank .* task 0 ' live_a_rng_smoke_dual_b_r1.log | sed 's/.*task 0 //' | sort -u)
rank_task1=$(grep '\[DualHead\] rank .* task 1 ' live_a_rng_smoke_dual_b_r1.log | sed 's/.*task 1 //' | sort -u)
if [ "$(printf '%s\n' "$rank_task0" | wc -l)" -ne 1 ] || [ "$(printf '%s\n' "$rank_task1" | wc -l)" -ne 1 ]; then
  echo "SMOKE FAIL: Dual-B calibration params differ across ranks"
  exit 1
fi
if ! grep -q '\[DualHead\] task 1 fused_proto_max_diff=.* PASS' live_a_rng_smoke_dual_b_r1.log; then
  echo "SMOKE FAIL: final lambda=1 fused/prototype logit check missing"
  exit 1
fi

echo "SMOKE PASS"
