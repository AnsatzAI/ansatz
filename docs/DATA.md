# Data provenance

## SQuADDS database (real Ansys Q3D simulations)

Tier 1 trains on the **SQuADDS** database (Shanto et al., *Quantum* 8, 1465
(2024)), HuggingFace dataset `SQuADDS/SQuADDS_DB`, **MIT license**:

- Config `qubit-TransmonCross-cap_matrix`: **1,934** designs, each an Ansys
  Q3D-extracted capacitance matrix (fF) for a TransmonCross (xmon) qubit with
  readout claw, simulated with adaptive refinement to 0.1% convergence
  (`percent_error: 0.1`, up to 30 passes) by the Levenson-Falk lab (USC).
- Varied parameters and observed ranges: `cross_length` 90–420 µm,
  `claw_length` 70–400 µm, `ground_spacing` 4.1–10 µm. Held fixed in the DB:
  `cross_width` = `cross_gap` = 30 µm, `claw_width` = 15 µm,
  `claw_gap` = 5.1 µm.
- Entries used: cross_to_ground, claw_to_ground, cross_to_claw, claw_to_claw,
  ground_to_ground.

Run `python scripts/fetch_squadds.py` to reproduce
`data/squadds_qubit_cap.parquet`. Please cite SQuADDS if you use this data.

## Generated field dataset (solver benchmarks + Tier 2 surrogate)

`python scripts/gen_fields.py` samples designs from ranges anchored to the DB
(the three DB-varied parameters use the DB ranges; DB-fixed parameters get
symmetric spreads around their standard values, widening the family), then
rasterizes and solves both conductor excitations to machine precision with a
sparse direct factorization:

| split       | n=255 | n=511 | n=1023 | n=2047 |
|-------------|------:|------:|-------:|-------:|
| train       | 2200  | 800   | 150    | —      |
| val         | 200   | 100   | —      | —      |
| test        | 200   | 100   | 120    | 30     |
| ood         | 150   | 75    | 60     | —      |
| routertrain | 150   | 80    | 50     | 20     |

OOD splits widen every parameter interval by 1.3x about its midpoint.
Shards are compressed npz (packed masks + float32 fields), ~200 MB total,
regenerable deterministically from fixed seeds.
