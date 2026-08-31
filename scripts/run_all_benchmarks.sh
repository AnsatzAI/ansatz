#!/bin/bash
# Full benchmark suite. Run after surrogate training produces runs/unet_255.pt.
# HINTS is only run at n<=511 (its fixed schedule is non-competitive and slow
# at larger n; measured and reported at the sizes where it is credible).
set -euo pipefail
cd "$(dirname "$0")/.."

W=runs/unet_255.pt

for SPLIT in routertrain test ood; do
  for N in 255 511; do
    if [ "$SPLIT" = "ood" ] && [ "$N" = 2047 ]; then continue; fi
    python scripts/benchmark.py --split $SPLIT --n $N --tol 1e-8 --weights $W \
      --pipelines direct,amg_cg,surr_cg,surr_amg,hints \
      --out runs/bench_${SPLIT}_${N}.parquet
  done
done

for SPLIT in routertrain test ood; do
  for N in 1023 2047; do
    if [ "$SPLIT" = "ood" ] && [ "$N" = 2047 ]; then continue; fi
    python scripts/benchmark.py --split $SPLIT --n $N --tol 1e-8 --weights $W \
      --pipelines direct,amg_cg,surr_cg,surr_amg \
      --out runs/bench_${SPLIT}_${N}.parquet
  done
done

python scripts/train_router.py \
  --train runs/bench_routertrain_255.parquet runs/bench_routertrain_511.parquet \
          runs/bench_routertrain_1023.parquet runs/bench_routertrain_2047.parquet \
  --eval  runs/bench_test_255.parquet runs/bench_test_511.parquet \
          runs/bench_test_1023.parquet runs/bench_test_2047.parquet \
  --out runs

python scripts/train_router.py \
  --train runs/bench_routertrain_255.parquet runs/bench_routertrain_511.parquet \
          runs/bench_routertrain_1023.parquet runs/bench_routertrain_2047.parquet \
  --eval  runs/bench_ood_255.parquet runs/bench_ood_511.parquet \
          runs/bench_ood_1023.parquet \
  --out runs/ood

echo "ALL BENCHMARKS DONE"
