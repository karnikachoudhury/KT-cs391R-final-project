import os
from tensorboard.backend.event_processing import event_accumulator

base_dir = "./.karnika_messing"

for run_name in os.listdir(base_dir):
    run_path = os.path.join(base_dir, run_name)
    if not os.path.isdir(run_path):
        continue

    print(f"\n=== {run_name} ===")
    try:
        ea = event_accumulator.EventAccumulator(run_path)
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        for tag in tags:
            print(tag)
    except Exception as e:
        print("Could not read:", e)