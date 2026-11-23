#!/usr/bin/env bash
# Run the four pore-charging cases matched to Pawel's DNS, one at a time, on the GPU.
# Plan: docs/pore_dns_plan.md, answers: docs/pore_dns_answers.md.
# SDL: sdl_radius=0 / sdl_length=0 select the defaults 2 max(alpha) and L/2 + 1.5 max(alpha),
# which are the DNS rules (narrow: 2 and 501.5; pores with max alpha = 2: 4 and 503).
# If a case fails (non-zero exit, e.g. out of memory), it is retried once on a coarser
# mesh (h_wall = COARSE_H, h_entr = h_wall/2) as results/pore_dns_<case>_coarse.
# Usage: bash tools/run_pore_dns.sh [case ...]     (default: narrow converging diverging wide)
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pnpflow

COARSE_H=${COARSE_H:-0.15}
DEVICE=${DEVICE:-cuda}
LOG=results/pore_dns_queue.log
mkdir -p results

declare -A A_IN=([narrow]=1 [converging]=2 [diverging]=1 [wide]=2)
declare -A A_OUT=([narrow]=1 [converging]=1 [diverging]=2 [wide]=2)

COMMON=(--set L_nm=10000 --set a_ref_nm=10
        --set phi_w_mV=10 --set c0_mM=0.948 --set eps=7.0832e-10 --set T=298.15 --set D=1.337e-9
        --set phi_end_mV=10 --set sdl_radius=0 --set sdl_length=0
        --set tau_end=1.01 --set dns_times=1 --set dns_init=1)

run_case() {  # name, extra args...
    local name=$1; shift
    mkdir -p "results/$name"
    echo "$(date '+%F %T') start $name $*" | tee -a "$LOG"
    local t0=$SECONDS
    JAX_PLATFORMS=$DEVICE python -u run.py pore_charging --name "$name" "${COMMON[@]}" "$@" \
        > "results/$name/stdout.log" 2>&1
    local rc=$?
    # run.py exits 0 even when the solve fails: treat a missing "converged: True" as failure
    [ $rc -eq 0 ] && ! grep -q "^converged: True" "results/$name/stdout.log" && rc=3
    echo "$(date '+%F %T') end   $name rc=$rc $((SECONDS - t0))s" | tee -a "$LOG"
    return $rc
}

cases=("$@")
[ ${#cases[@]} -eq 0 ] && cases=(narrow converging diverging wide)
for c in "${cases[@]}"; do
    geo=(--set alpha_in=${A_IN[$c]} --set alpha_out=${A_OUT[$c]})
    if [ "${COARSE_ONLY:-0}" = 1 ]; then      # skip the default mesh (known not to fit)
        run_case "pore_dns_${c}_coarse" "${geo[@]}" --set h_wall=$COARSE_H || true
    elif ! run_case "pore_dns_$c" "${geo[@]}"; then
        echo "$(date '+%F %T') retry $c on coarse mesh h_wall=$COARSE_H" | tee -a "$LOG"
        run_case "pore_dns_${c}_coarse" "${geo[@]}" --set h_wall=$COARSE_H || true
    fi
done
echo "$(date '+%F %T') queue done" | tee -a "$LOG"
