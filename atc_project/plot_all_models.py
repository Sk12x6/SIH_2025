# plot_all_models.py
import os
import subprocess

# Ensure output folder
outdir = "plots"
os.makedirs(outdir, exist_ok=True)

# List of models with target names
models = [
    ("models/milk_productivity_model.pkl", "milk_productivity"),
    ("models/longevity_model.pkl", "longevity"),
    ("models/reproductive_efficiency_model.pkl", "reproductive_efficiency"),
    ("models/elite_dam_model.pkl", "elite_dam")
]

csv_path = "data/cow_data.csv"

# Run plot_utils.py for each model
for model_path, target in models:
    cmd = f'python plot_utils.py --model {model_path} --csv {csv_path} --target {target} --outdir {outdir}'
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
