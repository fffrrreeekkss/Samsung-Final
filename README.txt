Full-surface experiment
Run: .venv-experiment/bin/python scripts/run_experiment.py --patients 2
Default runs only patient 2. Other numbers require explicit selection.
Numbering: benchmark/experiment.py PATIENTS; 1 is historical patient_010.
Inputs: data/ground_truth (originals), data/reference (existing minimal canonicalization, lossless PLY).
Final output: experiments/results/*.ply and metrics.csv.
1_historical_* preserves the old small case without reevaluation. Distances use unknown native source units.
Failed methods have missing metrics and no fabricated reconstruction.
