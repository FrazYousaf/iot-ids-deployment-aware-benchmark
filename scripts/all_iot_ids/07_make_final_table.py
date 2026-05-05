import os
import glob
import pandas as pd

import config
from utils import ensure_dirs

ensure_dirs(config.RESULTS_DIR)

summary_files = sorted(glob.glob(os.path.join(config.RESULTS_DIR, "*_summary.csv")))

if not summary_files:
    raise FileNotFoundError(
        f"No summary CSV files found in {config.RESULTS_DIR}. "
        "Run the model scripts first."
    )

dfs = [pd.read_csv(f) for f in summary_files]
table = pd.concat(dfs, ignore_index=True)

# Sort for readability
order_cols = [
    "model",
    "feature_set",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "fp_per_10000_benign",
    "training_time_sec",
    "total_inference_time_sec",
    "mean_latency_ms_per_sample",
    "p95_latency_ms_per_sample",
    "p99_latency_ms_per_sample",
    "throughput_samples_per_sec",
    "model_size_mb",
]

table = table[order_cols]
table = table.sort_values(
    by=["feature_set", "macro_f1", "throughput_samples_per_sec"],
    ascending=[True, False, False],
).reset_index(drop=True)

output_path = os.path.join(config.RESULTS_DIR, "final_model_comparison.csv")
table.to_csv(output_path, index=False)

print("\nFinal comparison table:")
print(table.to_string(index=False))

print("\nSaved final comparison table to:", output_path)
