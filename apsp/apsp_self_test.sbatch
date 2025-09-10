#!/bin/bash
#SBATCH -J APSP_SelfTest
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o apsp_output_%j.log
#SBATCH --mem=32G

set -e

echo "================================================="
echo "                  APSP Self-Test                 "
echo "================================================="
echo "Job started on $(hostname) at $(date)"
echo ""

cd $HOME/hip_programming_contest/apsp

# Clean and build
make clean >/dev/null 2>&1 || true
echo "STEP 1: Compiling source code..."
make
echo "Compilation successful."

TOTAL_TIME=0.0
PASSED_COUNT=0
TEST_COUNT=$(ls -1q testcases/*.in | wc -l)

echo "STEP 2: Running test cases..."
echo "Found ${TEST_COUNT} test cases in testcases/"
echo "-------------------------------------------------"

MY_OUTPUT_DIR="my_outputs"
rm -rf "${MY_OUTPUT_DIR}"
mkdir -p "${MY_OUTPUT_DIR}"

for input_file in testcases/*.in; do
    base_name=$(basename "${input_file}" .in)
    golden_output_file="testcases/${base_name}.out"
    my_output_file="${MY_OUTPUT_DIR}/${base_name}.myout"

    echo -n "Running test [${base_name}]... "
    if [ ! -f "${golden_output_file}" ]; then
        echo "SKIPPED. Reason: Golden output file ${golden_output_file} not found."
        continue
    fi

    ( /usr/bin/time -f "%e" ./apsp "${input_file}" --timing > "${my_output_file}" ) 2> time_temp.txt
    # Display timing info in log and extract execution time
    cat time_temp.txt
    EXEC_TIME=$(tail -1 time_temp.txt)
    rm -f time_temp.txt

    if diff -u "${golden_output_file}" "${my_output_file}" >/dev/null 2>&1; then
        echo "PASS (${EXEC_TIME}s)"
        PASSED_COUNT=$((PASSED_COUNT + 1))
        TOTAL_TIME=$(echo "${TOTAL_TIME} + ${EXEC_TIME}" | bc)
    else
        echo "FAIL"
        echo "-------------------------------------------------"
        echo "ERROR: Output mismatch on test case [${base_name}]."
        echo "Your output is in: ${my_output_file}"
        echo "The correct output is in: ${golden_output_file}"
        echo "You can use 'diff -u ${golden_output_file} ${my_output_file}' to see the difference."
        exit 1
    fi
done

echo "-------------------------------------------------"
echo "FINAL RESULT: ALL TESTS PASSED!"
echo ""
echo "    Passed cases: ${PASSED_COUNT} / ${TEST_COUNT}"
echo "    Total execution time: ${TOTAL_TIME} seconds"
echo ""
echo "Job finished at $(date)"
echo "================================================="


