# Ansatz 3D EM experiments — AWS handoff runbook

**Audience:** an agent with a large AWS instance, tasked with running the
Ansatz Rung-1 / Rung-2 experiments at industrially relevant scale.
**Repo:** https://github.com/AnsatzAI/ansatz (branch `main`; put AWS results
on a branch `aws-campaign`). Everything below is reproducible from the repo
plus two external builds (AWS Palace, Julia + DeviceLayout.jl).

---

## 0. Context in five sentences

Ansatz accelerates the electromagnetic design loop for superconducting
qubits. A 2D electrostatics benchmark showed sparse-direct solvers dominate
there (documented negative result, `docs/BENCHMARKS.md`), so the program moved
to **3D full-wave eigenmode simulation** (AWS Palace), where classical
practice genuinely hurts (hours per design, days per chip). The product
mechanism is **verified inverse design**: a forward model trained on our own
Palace campaigns chooses a geometry, a full Palace solve verifies it, and at
most one physics-based correction step follows — versus classical
optimization that spends a full solve per objective evaluation. At dev scale
on a laptop this gave **6.2x faster and 22.8x closer-to-spec** designs
(`docs/EM3D.md`). Your job: reproduce that at production fidelity (Rung 1)
and on a two-qubit unit cell (Rung 2), with real statistics.

## 1. What already exists (do not redo)

| Item | Status | Where |
|---|---|---|
| Palace transmon baseline reproduced (4.0991/5.6033 GHz, Q 18.5k) | done | `docs/EM3D.md` |
| Shift-cliff + config-frontier experiments | done | `runs/em3d_results.parquet` |
| Coarse single-cell campaign (101 solves, 5 params) | done | `runs/em3d_dataset.parquet` |
| Tier-1 3D forward model (LOO 0.88% f0 / 0.93% f1) | done | `runs/em3d_forward.pkl` |
| Dev-scale advantage demo (6.2x / 22.8x) | done | `runs/em3d_advantage.json` |
| Two-qubit unit-cell generator validated (4 modes correct) | done | `scripts/gen_two_transmon.jl` |
| Two-qubit campaign (n=40, laptop) | **running locally** (seed 7) | `runs/em3d_two_dataset.parquet` |
| Fine tier (order-3 + AMR-2) plumbing, delta trainer, `--fine` demo | written, **unrun** | scripts below |

The laptop orchestrator (`scripts/run_rungs.sh`) will produce small-scale
versions of everything in Sections 4–5. **Use different seeds** (>= 100) so
datasets merge without tag collisions.

## 2. Environment setup (Linux, x86_64 or arm64)

Instance guidance: eigenmode solves are memory- and core-bound. Good choices:
`c7a.16xlarge` (64 vCPU / 128 GB) for campaigns; `r7a.16xlarge` or
`hpc7a.48xlarge` for order-3 + AMR solves (peak RSS of a single fine solve
is expected in the 10–40 GB range at this mesh; measure it in E3 first).
200 GB gp3 EBS. Ubuntu 22.04 assumed below.

```bash
# system deps
sudo apt-get update && sudo apt-get install -y build-essential cmake gfortran \
    libopenmpi-dev openmpi-bin libopenblas-dev git pkg-config
# Julia (1.10+)
curl -fsSL https://install.julialang.org | sh -s -- -y && source ~/.bashrc
# Python
curl -fsSLo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh
bash miniconda.sh -b -p ~/miniconda3 && source ~/miniconda3/etc/profile.d/conda.sh
conda create -n ansatz python=3.11 -y && conda activate ansatz

# layout: scripts hard-code this tree (Path.home()/Documents/Personal/ansatz/...)
mkdir -p ~/Documents/Personal/ansatz && cd ~/Documents/Personal/ansatz
git clone https://github.com/AnsatzAI/ansatz.git
git clone --depth 1 https://github.com/awslabs/palace.git

# Palace superbuild (~30-60 min on 16+ cores)
cd palace && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$HOME/Documents/Personal/ansatz/palace-install \
      -DBLA_VENDOR=OpenBLAS
make -j $(nproc)
export PALACE_BIN=$HOME/Documents/Personal/ansatz/palace-install/bin/palace  # runner.py reads this

# DeviceLayout.jl environment (the transmon example ships its Project.toml)
cd ~/Documents/Personal/ansatz/palace/examples/transmon
julia --project -e 'using Pkg; Pkg.instantiate()'   # 10-25 min (precompile)

# ansatz package
cd ~/Documents/Personal/ansatz/ansatz
pip install -e ".[dev]" && pytest tests -q
```

Smoke tests (all must pass before campaigns):
```bash
cd ~/Documents/Personal/ansatz/ansatz
python scripts/em3d_experiments.py --exp baseline --ranks 16      # ~4.1 / 5.6 GHz
python scripts/em3d_campaign.py --n 1 --ranks 16 --seed 999       # one variant end-to-end
(cd ~/Documents/Personal/ansatz/palace/examples/transmon && \
  julia --project ~/Documents/Personal/ansatz/ansatz/scripts/gen_two_transmon.jl probe)
# expect GROUPS containing port_1, port_2, lumped_element_1, lumped_element_2
```

Path notes: `scripts/em3d_campaign.py` and `scripts/em3d_two_campaign.py`
define `EX = Path.home()/"Documents/Personal/ansatz/palace/examples/transmon"`;
mirror that layout or edit `EX`. Julia scripts are run with `cwd=EX`.

## 3. The physics and the knobs (what you are measuring)

- **Geometry family:** DeviceLayout `SingleTransmon` (xmon-style pad +
  claw-coupled quarter-wave readout on a feedline) and our `TwoTransmon`
  adaptation (two such pairs on one feedline, hangers at 1/4 and 3/4).
- **Solve:** Palace eigenmode, curl-curl Maxwell, sapphire substrate, junction
  as lumped L=14.86 nH / C=5.5 fF port. Outputs `eig.csv` (Re f, Q).
- **Fidelity tiers:** *coarse* = order 2, no AMR (labels; ~3–40 min on 6
  laptop cores); *fine/production* = order 3 + 2 AMR iterations
  (`transmon_amr.json` settings; the sign-off tier).
- **Two measured failure modes of classical practice** (keep them in mind):
  1. Eigensolver `Target` placed *above* a mode skips it at 2.5–3x cost;
     placed far below, the lossy readout mode (Q~8k) often fails to converge
     and silently vanishes (61% of far-target solves). Fix used everywhere:
     target = predicted f_q − 0.15…0.30 GHz, `N`=6–8, `MaxSize`=60–80, and a
     mode-completeness check (`identify_modes` / `_pick_modes`).
  2. Gradient-boosted forward models are piecewise constant — **never
     finite-difference them**. Corrections use quarter-wave 1/L scaling for
     resonators and local data-fit slopes for pads, damped (0.8) with trust
     regions.

## 4. Experiments — Rung 2 (two-qubit unit cell, frequency planning)

### E1. Two-qubit campaign (bespoke training data)
```bash
python scripts/em3d_two_campaign.py --n 400 --ranks 16 --seed 101
```
- Parameters (8 continuous, ordered so mode attribution is unambiguous):
  `cap_length_1 > cap_length_2 + 40` ⇒ f_q1 < f_q2; `total_length_1 ∈
  [4800,5400] > total_length_2 ∈ [4000,4600]` ⇒ f_r1 < f_r2. Ranges in
  `RANGES2`. Labels: `f_q1, f_q2, f_r1, f_r2` (+ `f_all`).
- Label-tier solver settings are set inside `main()` (`target 3.4, N=6,
  MaxSize=60, Tol=1e-6`). Resumable by tag; safe to run several instances
  in parallel **with different seeds** (each writes to the same parquet via
  read-append — run them from separate clones or serialize; simplest:
  one process with big `--n`, or copies of the repo per seed then merge).
- Scale target: ≥300 usable rows. Runtime: ~40 min/solve on 6 laptop cores;
  expect 5–10 min at 16–32 ranks.
- Sanity: `corr(total_length_i, f_ri) ≈ −0.99`, `corr(cap_length_i, f_qi)
  ≈ −0.9`, all Q_q ~1.5–2e4, Q_r ~7–9e3.

### E2. Forward model + frequency-planning advantage demo
```bash
python scripts/train_em3d_two_forward.py           # LOO per target; writes em3d_two_forward.pkl
cd scripts
for T in "4.00,4.35,5.45,6.40" "3.90,4.50,5.30,6.60" "4.10,4.25,5.60,6.20" \
         "4.30,4.70,5.25,6.80" "3.80,4.20,5.70,6.10"; do
  python em3d_two_demo.py --targets $T --tol-mhz 40 --ranks 16 \
      --max-classical-solves 12 --seed 0 --out-name "two_adv_${T//,/_}.json"
done
```
- Each JSON has both arms: wall-clock, solve count, worst per-mode miss (MHz),
  measured modes, design. Success = worst miss ≤ tol.
- Report: per-target table + aggregate (median/geomean speedup, fraction of
  targets each arm reaches tolerance). Ansatz arm uses ≤2 solves; classical
  arm is Nelder-Mead with a fixed budget — also try budget 20 for fairness.

### E3 (stretch). Two-qubit at production fidelity
`run_variant_two(..., solver_order=3, amr_iterations=2, timeout_s=14400)`
is supported; the campaign `main()` does not pass it — add a `--fine` flag
(3 lines) or run a subset via a small driver like `em3d_fine_campaign.py`.
Then a delta model as in Rung 1 and the E2 demo with fine verification.

## 5. Experiments — Rung 1 (production fidelity, single cell)

### E4. AMR reference (calibration + first coarse/fine pair)
```bash
cd ~/Documents/Personal/ansatz/palace/examples/transmon
/usr/bin/time -v $PALACE_BIN -np 16 transmon_amr.json | tee ~/amr_ref.log
# record wall-clock, max RSS, DOF per AMR iteration, final f0/f1 vs coarse 4.0991/5.6033
```

### E5. Fine campaign (multi-fidelity pairs)
```bash
python scripts/em3d_fine_campaign.py --k 60 --ranks 16
```
Re-solves K geometries from the coarse campaign at order-3 + AMR-2 with a
safe target (coarse f0 − 0.3). Writes `runs/em3d_dataset_fine.parquet` with
`coarse_f0/f1` alongside fine `f0/f1`. Scale target: K=40–60 (laptop plan
was 10).

### E6. Delta model
```bash
python scripts/train_em3d_delta.py     # constant vs linear delta, LOO; writes em3d_delta.pkl
```
Report `raw_coarse_vs_fine_mae_mhz` (how wrong coarse labels are at sign-off
fidelity) and the chosen delta's LOO MAE. If K ≥ 40, also try a GBR delta.

### E7. Production-fidelity verified inverse design
```bash
cd scripts
for T in "4.20,5.45" "4.05,5.60" "4.40,5.30" "3.95,5.80" "4.30,5.50"; do
  python em3d_advantage_demo.py --targets $T --tol-mhz 25 --ranks 16 \
      --max-classical-solves 10 --fine --out-name "fine_adv_${T//,/_}.json"
done
```
`--fine` switches verification solves to order-3 + AMR-2 and applies the delta
model to predictions. Both arms run at identical fidelity.

## 6. Timing-honesty protocol (non-negotiable)

1. **Never run two timed arms concurrently**, and never run campaigns while
   a demo arm is timing. Campaign `t_solve` values are diagnostics only.
2. Identical Palace settings across arms within a demo (order, AMR, N, Tol,
   MaxSize, ranks). The Ansatz arm's advantage must come from *solve count*
   and *targeting*, not from cheaper solves.
3. Always report the verified miss from the final Palace solve, the number
   of Palace solves, ranks, and instance type. Wall-clock is end-to-end
   (meshing + solve + model time).
4. Do not compare absolute times across machines; compare ratios measured on
   the same machine, and note the machine.
5. Report failures: mode dropouts, fallback triggers, budget exhaustion.
   Zero-failure claims require the count of cells checked.

## 7. Known pitfalls (all hit once already)

- `transmon.jl`'s README says `n_meander_turns`; installed
  `SingleTransmon.single_transmon` uses `n_meander_turns` and `l_claw`
  (GitHub main renamed them) — verify with the probe if DeviceLayout updates.
- Passing the bare `lumped_element` group to
  `singlechip_solidmodel_target` alongside indexed ones creates overlapping
  boundary elements → Palace aborts ("cannot have multiple boundary
  elements"). Only indexed groups are retained now.
- `generate_transmon(mesh_filename=...)` already prefixes `mesh/`.
- `datasets`/HF not needed for 3D work. `runs/` is gitignored except the
  force-added result artifacts listed in Section 9.
- The eigensolver can return spurious 12–17 GHz modes instead of the readout;
  always band-identify modes (`QUBIT_BAND`, `READOUT_BAND`) and treat a
  missing mode as a failed solve, not as data.
- Campaign scripts advance the RNG even for skipped tags (resumability
  depends on it) — do not reorder sampling.
- macOS-only: `setsid` absent (use Python `start_new_session`); N/A on Linux.

## 8. Deliverables to hand back

On branch `aws-campaign`:
- `runs/em3d_two_dataset.parquet` (merged), `runs/em3d_two_forward.pkl`,
  `runs/em3d_two_forward_results.json`, `runs/two_adv_*.json`
- `runs/amr_ref.log`, `runs/em3d_dataset_fine.parquet`,
  `runs/em3d_delta.pkl`, `runs/em3d_delta_results.json`, `runs/fine_adv_*.json`
- A results section appended to `docs/EM3D.md` with: instance type, ranks,
  per-solve times per tier, LOO tables, and the two demo tables (per target
  + aggregate). Same honesty framing as the existing sections.
- Optional: figures via matplotlib (light palette in `scripts/make_figures.py`
  shows the house style).

## 9. File map

```
scripts/gen_variant.jl            single-cell mesh+config from JSON spec (fine tier via amr_iterations)
scripts/gen_two_transmon.jl       two-qubit cell generator; `probe` prints physical groups
scripts/em3d_campaign.py          single-cell coarse campaign (run_variant: fidelity knobs)
scripts/em3d_fine_campaign.py     Rung-1 fine re-solves of coarse geometries
scripts/em3d_two_campaign.py      Rung-2 campaign (run_variant_two, identify_modes)
scripts/train_em3d_forward.py     single-cell forward model (LOO)
scripts/train_em3d_delta.py       multi-fidelity delta model
scripts/train_em3d_two_forward.py two-qubit forward model (LOO)
scripts/em3d_advantage_demo.py    single-cell verified inverse design (--fine)
scripts/em3d_two_demo.py          two-qubit verified frequency planning
scripts/em3d_experiments.py       baseline / shift / frontier
scripts/run_rungs.sh              laptop orchestrator (sequential)
src/ansatz/em3d/{config,runner}.py  Palace config templating + timed runner/parser
docs/EM3D.md                      measured results so far (read first)
```
