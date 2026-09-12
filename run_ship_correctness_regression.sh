#!/usr/bin/env bash
set -u
set -o pipefail

mkdir -p ship-regression-logs

PASS=0
FAIL=0

run_one() {
    local label="$1"
    local test_name="$2"
    local log="ship-regression-logs/${label}.log"

    echo
    echo "================================================================"
    echo "[SHIP regression] ${label}"
    echo "[test] ${test_name}"
    echo "================================================================"

    cargo test -p poulpy-cpu-ref --lib --all-features \
      "${test_name}" -- --exact --nocapture 2>&1 | tee "${log}"
    local status=${PIPESTATUS[0]}

    if ! grep -Eq 'running [1-9][0-9]* test' "${log}"; then
        echo "[FAIL] ${label}: no test matched exact name  (log: ${log})"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ "${status}" -eq 0 ]; then
        echo "[PASS] ${label}"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] ${label}  (log: ${log})"
        FAIL=$((FAIL + 1))
    fi
}

# ----------------------------------------------------------------------
# Gate A: established/original SHIP end-to-end correctness.
# ----------------------------------------------------------------------
run_one \
  "fft64_legacy_ship_bootstrap" \
  "tests::ckks_tests::fft64_f64::ship_bootstrap"

# Complex end-to-end SHIP after the earlier shared-mask / omega2-permutation /
# dual H-MUX + tensor-key optimization.  This protects the optimization stack
# that predates the new packed/sheared path.
run_one \
  "fft64_shared_complex_ship_bootstrap" \
  "tests::ckks_tests::fft64_f64::ship_bootstrap_complex"

# ----------------------------------------------------------------------
# Gate B: latest Packed/sheared-SHIP dataflow regression.
#
# This test currently covers packed H-MUX, sheared routing, full BRotMux,
# encrypted BRotMCol and the encrypted product tree.  Before the tensor fix,
# FFT64 is expected to fail at Phase 5C2b width=8.
# ----------------------------------------------------------------------
run_one \
  "fft64_packed_sheared_ship" \
  "tests::ckks_tests::fft64_f64::ship_mux_rotate"

# ----------------------------------------------------------------------
# Gate C: exact-backend controls.  These distinguish an FFT64-specific
# regression from an algorithm/dataflow regression.
# ----------------------------------------------------------------------
run_one \
  "ntt4x30_legacy_ship_bootstrap" \
  "tests::ckks_tests::ntt4x30_f128::ship_bootstrap"

run_one \
  "ntt4x30_shared_complex_ship_bootstrap" \
  "tests::ckks_tests::ntt4x30_f128::ship_bootstrap_complex"

run_one \
  "ntt4x30_packed_sheared_ship" \
  "tests::ckks_tests::ntt4x30_f128::ship_mux_rotate"

echo
echo "================================================================"
echo "[SHIP regression summary] PASS=${PASS} FAIL=${FAIL}"
echo "Logs: ship-regression-logs/"
echo "================================================================"

if [ "${FAIL}" -ne 0 ]; then
    exit 1
fi
