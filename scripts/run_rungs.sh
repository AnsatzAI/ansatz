#!/bin/bash
# Detached orchestrator for Rung 2 (two-qubit cell) then Rung 1 (production
# fidelity). Sequential so demo timings are clean. Emits PHASE markers.
# Launch:  setsid nohup caffeinate -i bash scripts/run_rungs.sh > runs/rungs.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ansatz
R=${RANKS:-6}
EX=~/Documents/Personal/ansatz/palace/examples/transmon
PAL=~/Documents/Personal/ansatz/palace-install/bin/palace
phase() { echo "PHASE $(date '+%F %T') $*"; }

phase START rungs

# ---------------- Rung 2: two-qubit unit cell ----------------
phase RUNG2 campaign start
python scripts/em3d_two_campaign.py --n 40 --ranks $R --seed 7 || phase ERROR two_campaign
phase RUNG2 campaign done
python scripts/train_em3d_two_forward.py || phase ERROR two_forward
phase RUNG2 forward done
( cd scripts && python em3d_two_demo.py --targets 4.00,4.35,5.45,6.40 --tol-mhz 40 \
    --ranks $R --max-classical-solves 10 ) || phase ERROR two_demo
phase RUNG2 demo done

# ---------------- Rung 1: production fidelity ----------------
phase RUNG1 amr_reference start
( cd "$EX" && /usr/bin/time -l "$PAL" -np $R transmon_amr.json > "$OLDPWD/runs/amr_ref.log" 2>&1 ) \
  || phase ERROR amr_reference
phase RUNG1 amr_reference done
python scripts/em3d_fine_campaign.py --k 10 --ranks $R || phase ERROR fine_campaign
phase RUNG1 fine campaign done
python scripts/train_em3d_delta.py || phase ERROR delta
phase RUNG1 delta done
( cd scripts && python em3d_advantage_demo.py --targets 4.20,5.45 --tol-mhz 25 \
    --ranks $R --max-classical-solves 8 --fine --out-name em3d_advantage_fine.json ) \
  || phase ERROR fine_demo
phase RUNG1 demo done

phase ALL_DONE
