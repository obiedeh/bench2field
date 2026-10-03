#!/bin/bash
# Temporary CPU governor test for case study 01 (phase 2). SPEED ONLY,
# ACCURACY NOT YET MEASURED.
#
# Sets every cpufreq policy to the `performance` governor, runs three
# alternated pipeline-profile pairs (fp32, fp16) on the 480p rover frames,
# and restores the governor and limits it found, even if a run fails. The
# settings before, during and after are written into the run folder. Nothing
# persistent is changed: no nvpmodel mode, no boot-time setting; a reboot
# would undo it too.
#
# Needs root for the sysfs writes; the profiles run as the invoking user.
#   cd ~/github/bench2field && sudo bash case_studies/01_perception_detector/governor_test.sh <commit>
set -u
COMMIT="${1:?usage: sudo bash governor_test.sh <expected commit>}"
RUN_AS="${SUDO_USER:?run this with sudo from the account that owns the checkout}"
[ "$(id -u)" = "0" ] || { echo "needs sudo for the sysfs writes"; exit 1; }
cd "$(dirname "$0")/../.." || exit 1
OUT=case_studies/01_perception_detector/runs/phase2_orin_profiles_governor
POLICIES=(/sys/devices/system/cpu/cpufreq/policy*)
sudo -u "$RUN_AS" mkdir -p "$OUT"

snapshot() {  # governor and limits of every policy, one line each
    for p in "${POLICIES[@]}"; do
        echo "$(basename "$p") cpus=[$(cat "$p/affected_cpus")] governor=$(cat "$p/scaling_governor") min_khz=$(cat "$p/scaling_min_freq") max_khz=$(cat "$p/scaling_max_freq")"
    done
}

declare -A GOV MIN MAX
for p in "${POLICIES[@]}"; do
    GOV[$p]=$(cat "$p/scaling_governor"); MIN[$p]=$(cat "$p/scaling_min_freq"); MAX[$p]=$(cat "$p/scaling_max_freq")
done
{ echo "# before, $(date -u +%Y-%m-%dT%H:%M:%SZ)"; snapshot; echo "# nvpmodel: $(nvpmodel -q 2>/dev/null | head -1)"; } > "$OUT/governor_before.txt"

restore() {
    for p in "${POLICIES[@]}"; do
        echo "${GOV[$p]}" > "$p/scaling_governor"
        echo "${MIN[$p]}" > "$p/scaling_min_freq"
        echo "${MAX[$p]}" > "$p/scaling_max_freq"
    done
    { echo "# after restore, read back $(date -u +%Y-%m-%dT%H:%M:%SZ)"; snapshot; echo "# nvpmodel: $(nvpmodel -q 2>/dev/null | head -1)"; } > "$OUT/governor_after.txt"
    if diff <(grep -v '^#' "$OUT/governor_before.txt") <(grep -v '^#' "$OUT/governor_after.txt") > /dev/null; then
        echo "# RESTORED: settings read back equal to the ones recorded before" >> "$OUT/governor_after.txt"
    else
        echo "# MISMATCH: settings after restore differ from before" >> "$OUT/governor_after.txt"
    fi
    chown "$RUN_AS": "$OUT"/governor_*.txt
    tail -1 "$OUT/governor_after.txt"
}
trap restore EXIT

for p in "${POLICIES[@]}"; do echo performance > "$p/scaling_governor"; done
{ echo "# during the test, $(date -u +%Y-%m-%dT%H:%M:%SZ)"; snapshot; } > "$OUT/governor_during.txt"
cat "$OUT/governor_during.txt"

for r in 1 2 3; do
    for prec in fp32 fp16; do
        echo "== $prec r$r $(date +%H:%M:%S)"
        sudo -u "$RUN_AS" .venv/bin/python case_studies/01_perception_detector/profile_pipeline.py \
            --frames data/rover_frames_480p --precision "$prec" --no-spin --expect-commit "$COMMIT" \
            --out "$OUT/${prec}_r$r.json" 2>&1 | grep -E "^total|^cpu frequency|expected commit|Error"
    done
done
