#!/usr/bin/env bash
set +e

PROJECT_ROOT="/home/zhaoyang/SD-Lora-CL"
CONDA_SH="/home/zhaoyang/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sdlora"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC="${NPROC:-4}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

cd "$PROJECT_ROOT" || exit 1
source "$CONDA_SH" || exit 1
conda activate "$CONDA_ENV" || exit 1
export HF_ENDPOINT
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1

if [ -n "$(git status --porcelain)" ]; then
  echo "P3 FAIL: working tree is not clean; commit before launching"
  exit 1
fi

run_one() {
  local name="$1" config="$2" outdir="$3" logfile="$4"
  local lock="${outdir%/}.lock"
  if [ -e "$outdir" ]; then
    echo "P3 FAIL: $name output directory already exists: $outdir"
    exit 1
  fi
  if ! flock -n "$lock" true 2>/dev/null; then
    echo "P3 FAIL: $name filepath lock is held: $lock"
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
    echo "P3 FAIL: $name exited with status $status"
    exit 1
  fi
  if grep -qiE 'non-determin|warn_only=True|memory efficient attention defaults to a non' "$logfile"; then
    echo "P3 FAIL: $name log contains a determinism warning"
    exit 1
  fi
}

for dataset in inr c100; do
  for seed in 1 2 3 4 5; do
    for method in livea_dual_b sdlora exp009; do
      if [ "$dataset" = "inr" ]; then
        ds_upper="INR"
      else
        ds_upper="C100"
      fi
      case "$method" in
        livea_dual_b) method_upper="LIVEA_DUALB" ;;
        sdlora) method_upper="SDLORA" ;;
        exp009) method_upper="EXP009" ;;
      esac
      outdir="./${ds_upper}_P3_${method_upper}_SEED${seed}_NCCL/"
      config="exps/p3_${dataset}_${method}_seed${seed}_nccl.json"
      log="p3_${dataset}_${method}_seed${seed}_nccl.log"
      run_one "${ds_upper}_${method_upper}_SEED${seed}" "$config" "$outdir" "$log"
    done
  done
done

echo "===== P3 stats ====="
{
  echo "INR"
  for metric in final avg forgetting; do
    echo "## $metric"
    python scripts/multiseed_stats.py \
      --group dual "inr_p1_livea_dual_b_seed1995_nccl.log" \
      --group dual "p3_inr_livea_dual_b_seed*_nccl.log" \
      --group sdlora "inr_p1_sdlora_seed1995_nccl.log" \
      --group sdlora "p3_inr_sdlora_seed*_nccl.log" \
      --group exp009 "inr_p1_exp009_seed1995_nccl.log" \
      --group exp009 "p3_inr_exp009_seed*_nccl.log" \
      --metric "$metric" \
      --paired dual sdlora --margin 0.5
    python scripts/multiseed_stats.py \
      --group dual "inr_p1_livea_dual_b_seed1995_nccl.log" \
      --group dual "p3_inr_livea_dual_b_seed*_nccl.log" \
      --group exp009 "inr_p1_exp009_seed1995_nccl.log" \
      --group exp009 "p3_inr_exp009_seed*_nccl.log" \
      --metric "$metric" \
      --paired dual exp009 --margin 0.5
  done
  echo "C100"
  for metric in final avg forgetting; do
    echo "## $metric"
    python scripts/multiseed_stats.py \
      --group dual "c100_p2_livea_dual_b_seed1993_nccl.log" \
      --group dual "p3_c100_livea_dual_b_seed*_nccl.log" \
      --group sdlora "c100_p2_sdlora_seed1993_nccl.log" \
      --group sdlora "p3_c100_sdlora_seed*_nccl.log" \
      --group exp009 "c100_p2_exp009_seed1993_nccl.log" \
      --group exp009 "p3_c100_exp009_seed*_nccl.log" \
      --metric "$metric" \
      --paired dual sdlora --margin 0.5
    python scripts/multiseed_stats.py \
      --group dual "c100_p2_livea_dual_b_seed1993_nccl.log" \
      --group dual "p3_c100_livea_dual_b_seed*_nccl.log" \
      --group exp009 "c100_p2_exp009_seed1993_nccl.log" \
      --group exp009 "p3_c100_exp009_seed*_nccl.log" \
      --metric "$metric" \
      --paired dual exp009 --margin 0.5
  done
} > p3_multiseed_stats_output.txt 2>&1

echo "P3 QUEUE DONE"
