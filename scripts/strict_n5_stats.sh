#!/usr/bin/env bash
# Strict confirmation-seed statistics for the P3 frozen results.
#
# Confirmation seeds: ImageNet-R 1,2,3,4,5 and CIFAR-100 1,2,3,4,5.
# Development seeds 1995/1993 are intentionally excluded and are reported
# separately as descriptive sensitivity analysis (see p3_multiseed_stats_output.txt).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="p3_strict_n5_stats_output.txt"
> "$OUT"
PYTHON="${PYTHON:-python}"

run_stats() {
    local dataset="$1"
    "$PYTHON" scripts/multiseed_stats.py \
        --group dual "p3_${dataset}_livea_dual_b_seed*_nccl.log" \
        --group sdlora "p3_${dataset}_sdlora_seed*_nccl.log" \
        --group exp009 "p3_${dataset}_exp009_seed*_nccl.log" \
        --metric "$2" \
        --paired dual sdlora \
        --paired dual exp009 \
        --seeds 1,2,3,4,5 \
        --margin 0.5 \
        >> "$OUT"
}

{
    echo "Strict n=5 statistics (confirmation seeds 1-5 only; dev seeds 1995/1993 excluded)"
    echo "Generated: $(date +%F_%T)"
    echo
} >> "$OUT"

for metric in final avg forgetting; do
    {
        echo "INR (ImageNet-R) / $metric"
        echo "======================================================================"
    } >> "$OUT"
    run_stats inr "$metric"
    echo >> "$OUT"
done

for metric in final avg forgetting; do
    {
        echo "C100 (CIFAR-100) / $metric"
        echo "======================================================================"
    } >> "$OUT"
    run_stats c100 "$metric"
    echo >> "$OUT"
done

echo "strict n=5 stats written to $OUT"
