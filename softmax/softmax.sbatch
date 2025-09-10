#!/bin/bash
#SBATCH -J softmax_selftest
#SBATCH --partition=GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o softmax_%j.log
#SBATCH --mem=32G

set -euo pipefail

ROOT="/home/user095/hip_programming_contest/softmax"
cd "$ROOT"

echo "[INFO] Host: $(hostname)  Time: $(date)"
echo "[INFO] Build..."
make | cat

EXEC="./softmax"
if [ ! -x "$EXEC" ]; then
  echo "[ERROR] build failed: $EXEC not found or not executable"
  exit 1
fi

VERIFIER="$ROOT/verify.py"
OUTDIR="$ROOT/my_outputs"
GOLDEN="$ROOT/testcases"
mkdir -p "$OUTDIR"

TOTAL=0
PASS=0
TOTAL_TIME=0.0

while IFS= read -r in_file; do
  base=$(basename "$in_file" .in)
  golden="$GOLDEN/$base.out"
  myout="$OUTDIR/$base.myout"
  echo -n "[RUN] $base ... "
  /usr/bin/time -f "%e" $EXEC "$in_file" > "$myout" 2>time.tmp || { echo "RUN FAIL"; exit 1; }
  t=$(cat time.tmp)
  rm -f time.tmp
  if python3 "$VERIFIER" "$myout" "$golden" >/dev/null 2>&1; then
    echo "PASS (${t}s)"
    PASS=$((PASS+1))
  else
    echo "FAIL"
    echo "  diff -u $golden $myout"
    diff -u "$golden" "$myout" | cat || true
    exit 1
  fi
  # 浮点累加总时间（避免依赖 bc）
  TOTAL_TIME=$(awk -v a="$TOTAL_TIME" -v b="$t" 'BEGIN{printf "%.6f", a + b}')
  TOTAL=$((TOTAL+1))
done < <(printf "%s\n" "$ROOT"/testcases/*.in | sort -V)

echo "[SUMMARY] $PASS / $TOTAL passed"
echo "[TIME] Total time: ${TOTAL_TIME}s"
if [ "$PASS" -ne "$TOTAL" ]; then
  exit 1
fi
echo "[DONE] All tests passed."


