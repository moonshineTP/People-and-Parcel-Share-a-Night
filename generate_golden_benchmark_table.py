"""
Generate benchmark results CSV with best costs and gap percentages for each solver.
Supports multiple datasets: Golden, Li, etc.
"""

import pandas as pd
import re
import sys
from pathlib import Path

# Get dataset from command line argument (default: Golden)
dataset = sys.argv[1].strip() if len(sys.argv) > 1 else "Golden"
dataset = dataset.capitalize()  # Ensure proper case

# Paths
attempts_csv = f"share_a_ride/data/benchmark/{dataset}/{dataset}-attempts.csv"
scoreboard_csv = f"share_a_ride/data/benchmark/{dataset}/{dataset}-scoreboard.csv"
output_csv = f"share_a_ride/data/benchmark/{dataset}/{dataset}_results.csv"

print(f"Processing {dataset} dataset...")

# Step 1: Load attempts data first to find best costs per solver-instance
df_attempts = pd.read_csv(attempts_csv)
df_attempts = df_attempts[df_attempts["dataset"] == dataset].copy()

# Keep all attempts with valid costs (accept 'done', 'overtime', etc.)
df_attempts = df_attempts.dropna(subset=["cost"])
df_attempts["cost"] = df_attempts["cost"].astype(float)

# Step 1b: Load scoreboard to include best_solver data
df_scoreboard = pd.read_csv(scoreboard_csv)
df_scoreboard = df_scoreboard[["instance", "best_cost", "best_solver"]].copy()
df_scoreboard.columns = ["instance", "scoreboard_cost", "scoreboard_solver"]

# Merge attempts with scoreboard to have complete picture
df = df_attempts.copy()

# Get unique instances from the data
unique_instances = sorted(df_attempts["instance"].unique())

# Calculate best cost across all solvers for each instance (to use as reference)
best_costs_per_instance = df.groupby("instance")["cost"].min().to_dict()

# Also merge in scoreboard best costs
scoreboard_costs = df_scoreboard.set_index("instance")["scoreboard_cost"].to_dict()

print(
    f"Best found cost from attempts + scoreboard per instance ({len(unique_instances)} instances):"
)
for instance in unique_instances:
    attempts_best = best_costs_per_instance.get(instance)
    scoreboard_best = scoreboard_costs.get(instance)
    overall_best = min(
        [x for x in [attempts_best, scoreboard_best] if x is not None], default=None
    )
    best_costs_per_instance[instance] = overall_best
    print(
        f"  {instance}: attempts={attempts_best}, scoreboard={scoreboard_best} -> best={overall_best}"
    )

# Step 2: Get best cost per solver-instance combination from attempts
best_costs = df.groupby(["instance", "solver"])["cost"].min().reset_index()
best_costs.columns = ["instance", "solver", "best_cost"]

# Also add scoreboard data as a row for each instance
scoreboard_rows = []
for _, row in df_scoreboard.iterrows():
    scoreboard_rows.append(
        {
            "instance": row["instance"],
            "solver": row["scoreboard_solver"],
            "best_cost": row["scoreboard_cost"],
        }
    )

if scoreboard_rows:
    df_scoreboard_costs = pd.DataFrame(scoreboard_rows)
    # Take the minimum cost between attempts and scoreboard for each solver-instance pair
    best_costs = pd.concat([best_costs, df_scoreboard_costs], ignore_index=True)
    best_costs = (
        best_costs.groupby(["instance", "solver"])["best_cost"].min().reset_index()
    )

# Step 3: Create pivot table structure
solvers = sorted(best_costs["solver"].unique())

print(f"\nSolvers found: {solvers}")
print(f"Instances ({len(unique_instances)}): {unique_instances}")

# Create result dataframe
result_rows = []

for instance in unique_instances:
    row = {"Instance": instance}
    instance_data = best_costs[best_costs["instance"] == instance]
    best_found = best_costs_per_instance.get(instance)

    for solver in solvers:
        solver_data = instance_data[instance_data["solver"] == solver]

        if len(solver_data) > 0:
            cost = solver_data["best_cost"].values[0]

            if best_found is not None:
                gap = ((cost - best_found) / best_found) * 100
                row[f"{solver}_cost"] = cost
                row[f"{solver}_gap"] = round(gap, 2)
            else:
                row[f"{solver}_cost"] = cost
                row[f"{solver}_gap"] = "--"
        else:
            row[f"{solver}_cost"] = "--"
            row[f"{solver}_gap"] = "--"

    row["Best_found"] = best_found
    result_rows.append(row)

result_df = pd.DataFrame(result_rows)

# Step 4: Sort rows by instance number (extract numeric part from instance name)
result_df["instance_num"] = result_df["Instance"].apply(
    lambda x: int(x.split("_")[1]) if "_" in x else float("inf")
)
result_df = (
    result_df.sort_values("instance_num")
    .drop("instance_num", axis=1)
    .reset_index(drop=True)
)

# Step 5: Add average gap row
avg_gap_row = {"Instance": "Avg gap"}

avg_gaps = {}
for solver in solvers:
    gap_col = f"{solver}_gap"
    gaps = [v for v in result_df[gap_col] if isinstance(v, (int, float))]

    if gaps:
        avg_gap = round(sum(gaps) / len(gaps), 2)
        avg_gaps[solver] = avg_gap
        avg_gap_row[gap_col] = avg_gap
    else:
        avg_gaps[solver] = float("inf")
        avg_gap_row[gap_col] = "--"

    avg_gap_row[f"{solver}_cost"] = ""

avg_gap_row["Best_found"] = ""

result_df = pd.concat([result_df, pd.DataFrame([avg_gap_row])], ignore_index=True)

# Step 6: Reorder columns by average gap (ascending)
# Sort solvers by average gap
sorted_solvers = sorted(solvers, key=lambda x: avg_gaps.get(x, float("inf")))

# Build new column order
new_columns = ["Instance"]
for solver in sorted_solvers:
    new_columns.append(f"{solver}_cost")
    new_columns.append(f"{solver}_gap")
new_columns.append("Best_found")

result_df = result_df[new_columns]

# Step 7: Save to CSV
result_df.to_csv(output_csv, index=False)
print(f"\nResults saved to {output_csv}")
print(f"\nPreview (first 5 rows + avg row):")
print(result_df.to_string())
