"""Run the full preprocessing pipeline for all currency pairs."""
from ml.data_pipeline import load_all, run_pipeline

data = load_all()
for name, df in data.items():
    run_pipeline(df, pair_name=name, sequence_length=60)
