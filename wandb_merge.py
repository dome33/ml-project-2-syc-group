import wandb
import pandas as pd

entity = "martinagallati-zhaw"  # W&B username or org
project = "falconet-new-pretraining"
run_ids = [
    "ny8vxh46", "3uz2kyj2", "8z6c6yjl", "xqzva0xy", "l7jza3vf",
    "8cayeas0", "vx1jx5f4", "k705mh1m", "puskcms4", "rt0wkzvm",
    "m3utltft", "e2qjfmtq", "n92xzbhf", "3wpb1cew", "eywfebo9",
    "81t2zakm"
]

wandb.login()
api = wandb.Api()
all_rows = []

epoch_offset = 0  # So epochs continue smoothly across runs

for run_id in run_ids:
    print(run_id)
    run = api.run(f"{entity}/{project}/{run_id}")
    df = run.history(samples=1000000)
    print(df.head())

    # If no 'epoch' column, create one
    if 'epoch' not in df.columns:
        df['epoch'] = range(len(df))

    # Keep only last log per epoch (most common)
    df = df.groupby("epoch").last().reset_index()

    # Offset epochs so they don't overlap
    df["epoch"] = df["epoch"] + epoch_offset
    df["source_run"] = run_id

    print(epoch_offset)
    epoch_offset = df["epoch"].max() + 1
    print(epoch_offset)
    all_rows.append(df)

# === MERGE ALL AND LOG ===
merged_df = pd.concat(all_rows, ignore_index=True)
merged_df = merged_df.sort_values(by="epoch").reset_index(drop=True)
merged_df["log_step"] = range(len(merged_df))  # clean step counter

# Init stitched run
stitched = wandb.init(
    project=project,
    name="stitched-full-run-hugo28_test",
    notes="Merged from 14 offline runs, 1 point per epoch",
    config={"source_runs": run_ids}
)

for _, row in merged_df.iterrows():
    data = row.dropna().to_dict()
    step = int(row["log_step"])
    data.pop("log_step", None)
    stitched.log(data, step=step)

stitched.finish()

print(f"✅ Merged run uploaded: https://wandb.ai/{entity}/{project}/runs/{stitched.id}")
