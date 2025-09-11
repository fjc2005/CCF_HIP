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

echo "================================================="
echo "                 Softmax Self-Test               "
echo "================================================="
echo "[INFO] Host: $(hostname)  Time: $(date)"
echo ""

echo "[STEP 1] Clean & Build"
make clean >/dev/null 2>&1 || true
make | cat
echo "[INFO] Build done."

EXEC="./softmax"
if [ ! -x "$EXEC" ]; then
  echo "[ERROR] build failed: $EXEC not found or not executable"
  exit 1
fi

VERIFIER="$ROOT/verify.py"
OUTDIR="$ROOT/my_outputs"
GOLDEN="$ROOT/testcases"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

if [ ! -f "$VERIFIER" ]; then
  echo "[ERROR] verifier not found: $VERIFIER"
  exit 1
fi

TOTAL=0
PASS=0
TOTAL_TIME=0.0

TEST_COUNT=$(find "$GOLDEN" -maxdepth 1 -type f -name "*.in" | wc -l | awk '{print $1}')
echo "[STEP 2] Run test cases"
echo "[INFO] Found ${TEST_COUNT} test cases in $(realpath "$GOLDEN")"
echo "-------------------------------------------------"

if [ "$TEST_COUNT" -eq 0 ]; then
  echo "[WARN] No testcases found under $GOLDEN"
  echo "[SUMMARY] 0 / 0 passed"
  echo "[TIME] Total time: 0.0s"
  exit 0
fi

while IFS= read -r -d '' in_file; do
  base=$(basename "$in_file" .in)
  golden="$GOLDEN/$base.out"
  myout="$OUTDIR/$base.myout"
  echo -n "[RUN] $base ... "
  tfile=$(mktemp -p "$ROOT" softmax_time_XXXXXX)
  (/usr/bin/time -f "%e" $EXEC "$in_file" --timing > "$myout") 2>"$tfile" || { echo "RUN FAIL"; rm -f "$tfile"; exit 1; }
  t=$(tail -n 1 "$tfile")
  rm -f "$tfile"
  if [ ! -f "$golden" ]; then
    echo "SKIPPED. Reason: golden not found: $golden"
    continue
  fi
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
done < <(find "$GOLDEN" -maxdepth 1 -type f -name "*.in" -print0 | sort -zV)

echo "[SUMMARY] $PASS / $TOTAL passed"
echo "[TIME] Total time: ${TOTAL_TIME}s"
if [ "$PASS" -ne "$TOTAL" ]; then
  exit 1
fi
echo "[DONE] All tests passed."


