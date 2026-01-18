"""
Module to transform parsed data structures into DataFrames for visualization.

This module serves as the Transform layer in the ETL pipeline:
- Extract: parser.py (parses raw files into structured data)
- Transform: transformer.py (converts structured data into visualization-ready DataFrames)
- Load: loader.py (provides data access for the application layer)
"""
import json
import ast
import math
from typing import Any, Literal

import pandas as pd

from share_a_ride.data.classes import Dataset
from share_a_ride.data.router import path_router
from share_a_ride.data.extractor import extract_instance_names, extract_csv_dataframe
from share_a_ride.data.parser import (
    parse_sarp_content, parse_sol_content, InstanceContent, SolutionContent
)

# --- Type aliases ---
CatalogDF = pd.DataFrame            # Columns: [purpose, dataset, instance]
AttemptsDF = pd.DataFrame     # Columns: [attempt_id, timestamp, dataset, ...]
ScoreboardDF = pd.DataFrame   # Columns: [dataset, instance, num_attempts, ...]
NodesDF = pd.DataFrame              # Columns: [id, x, y, demand, type_id, type_label]
EdgesDF = pd.DataFrame              # Columns: [vehicle_id, order, x, y, ...]
OrientIndex: Literal["index"] = "index"


# --- Columns Definitions ---
CATALOG_COLUMNS = [
    "purpose",
    "dataset",
    "instance",
]

ATTEMPT_DF_COLUMNS = [
    "instance",
    "attempt_id",
    "timestamp",
    "status",    
    "solver",
    "seed",
    "time_limit",
    "hyperparams",
    "elapsed_time",
    "cost",
    "info",
    "note",
]

SCOREBOARD_DF_COLUMNS = [
    "dataset",
    "instance",
    "num_attempts",
    "successful_attempts",
    "best_cost",
    "best_attempt_id",
    "best_timestamp",
    "best_solver",
    "best_solver_args",
    "best_solver_hyperparams",
    "best_time_taken",
    "cost_improvement",
    "percentage_improvement",
    "notes",
]

NODE_DF_COLUMNS = [
    "id",
    "x",
    "y",
    "demand",
    "type_id",
    "type_label",
]

EDGE_DF_COLUMNS = [
    "vehicle_id",
    "order",
    "x",
    "y",
    "mid_x",
    "mid_y",
    "angle",
    "distance",
]




# ================ Catalog Transformation ================
def _build_catalog_df() -> CatalogDF:
    """
    Internal builder: Build catalog dataframe mapping catalog structure of the dataset,
    down to instance level. Returns a dataframe with the following columns:
    - purpose: purpose of the dataset (e.g., "test", "train", etc.)
    - dataset: name of the dataset
    - instance: instance identifier (NaN when none available)
    """
    # Define columns and rows
    columns = pd.Index(CATALOG_COLUMNS)
    rows: list[dict[str, str]] = []

    # Extract dataset and instance information
    dataset_info = sorted([
        (dts.value.purpose.value, dts.value.name) for dts in Dataset
    ])
    for purpose, dts_name in sorted(dataset_info):
        # Crawl all instance names
        dataset_root = path_router(dts_name, "readall")
        instance_names: list[str] = extract_instance_names(dataset_root)

        # Add rows for each instance in the dataset
        rows.extend(
            {
                "purpose": purpose,
                "dataset": dts_name,
                "instance": inst_name,
            }
            for inst_name in instance_names
        )

    # If no rows (no instances), return empty DataFrame with correct columns
    if not rows:
        return pd.DataFrame(columns=columns)

    # Else, build DataFrame
    catalog = (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["purpose", "dataset", "instance"], kind="mergesort")
        .reset_index(drop=True)
    )
    return catalog




# --- External APIs ---
def transform_catalog() -> CatalogDF:
    """
    External API: Build and return the dataset catalog DataFrame.
    
    Returns:
        CatalogDF with columns: [purpose, dataset, instance]
    """
    return _build_catalog_df()




# ================ Visualizer Transformation ================
def _build_nodes_df(content: InstanceContent) -> NodesDF:
    """
    Internal builder: Transform a InstanceContent dictionary into a nodes DataFrame.

    Node ordering follows canonical order (0-indexed internally):
    - depot (0) - single depot per instance
    - passenger pickups (1..N)
    - parcel pickups (N+1..N+M)
    - passenger dropoffs (N+M+1..2N+M)
    - parcel dropoffs (2N+M+1..2N+2M)

    Note: SARP files use 1-indexed nodes, but we convert to 0-indexed for internal use.
    The conversion is: internal_id = sarp_id - 1

    Returns:
        NodesDF with columns: [id, x, y, demand, type_id, type_label]
    """
    # Identify node IDs for each category (these are 1-indexed from SARP file)
    depot_id_sarp = content.get('depot', 1)  # Default depot is 1 in SARP format
    ppick_ids_sarp = [p[0] for p in content['passenger_pairs']]
    lpick_ids_sarp = [p[0] for p in content['parcel_pairs']]
    pdrop_ids_sarp = [p[1] for p in content['passenger_pairs']]
    ldrop_ids_sarp = [p[1] for p in content['parcel_pairs']]

    # Coords and Demands (keyed by 1-indexed SARP IDs)
    coords_dict = content['coords_dict']
    q_dict = content['parcel_quantities_dict']


    # //// Build data rows (converting to 0-indexed)
    # Helper to get coord safely (using 1-indexed SARP key)
    def get_xy(sarp_id):
        return coords_dict.get(sarp_id, (0.0, 0.0))

    # Data rows
    data = []

    # depot (type 0) - internal id 0
    x, y = get_xy(depot_id_sarp)
    data.append(
        {"id": 0, "x": x, "y": y, "demand": 0, "type_id": 0, "type_label": "Depot"}
    )

    # ppick (type 1) - internal ids 1..N
    for i, sarp_id in enumerate(ppick_ids_sarp):
        x, y = get_xy(sarp_id)
        internal_id = sarp_id - 1  # Convert to 0-indexed
        data.append(
            {"id": internal_id, "x": x, "y": y, "demand": 1, "type_id": 1, "type_label": "pickP"}
        )

    # lpick (type 3)
    for i, sarp_id in enumerate(lpick_ids_sarp):
        x, y = get_xy(sarp_id)
        internal_id = sarp_id - 1  # Convert to 0-indexed
        qty = q_dict.get(sarp_id, 0)
        data.append(
            {"id": internal_id, "x": x, "y": y, "demand": qty, "type_id": 3, "type_label": "pickL"}
        )

    # pdrop (type 2)
    for i, sarp_id in enumerate(pdrop_ids_sarp):
        x, y = get_xy(sarp_id)
        internal_id = sarp_id - 1  # Convert to 0-indexed
        data.append(
            {"id": internal_id, "x": x, "y": y, "demand": -1, "type_id": 2, "type_label": "dropP"}
        )

    # ldrop (type 4)
    for i, sarp_id in enumerate(ldrop_ids_sarp):
        x, y = get_xy(sarp_id)
        internal_id = sarp_id - 1  # Convert to 0-indexed
        pickup_sarp_id = lpick_ids_sarp[i] if i < len(lpick_ids_sarp) else sarp_id
        qty = q_dict.get(pickup_sarp_id, 0)
        data.append(
            {"id": internal_id, "x": x, "y": y, "demand": -qty, "type_id": 4, "type_label": "dropL"}
        )

    # //// Create DF
    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(columns=["id", "x", "y", "demand", "type_id", "type_label"])

    return df


def _build_edges_df(
        solution_data: SolutionContent,
        nodes_df: NodesDF,
        default_depot: int = 0
    ) -> tuple[EdgesDF, int | None]:
    """
    Internal builder: Transform a SolutionContent dictionary into an edges DataFrame.

    Args:
        solution_data: Parsed solution dict with 'cost' and 'routes' keys
        nodes_df: DataFrame with node coordinates (must have 'id', 'x', 'y' columns)
        default_depot: Default depot node ID to inject if missing from routes

    Returns:
        - DataFrame with columns: [vehicle_id, order, x, y, mid_x, mid_y, angle, distance]
        - Reported cost from solution file (or None)
    """
    edges: list[dict[str, Any]] = []

    # Build coordinate lookup from nodes DataFrame
    # Use explicit Python int keys to avoid numpy type mismatch issues
    coords: dict[int, dict[str, float]] = {}
    for _, row in nodes_df.iterrows():
        node_id = int(row["id"])
        coords[node_id] = {"x": float(row["x"]), "y": float(row["y"])}

    reported_cost = solution_data.get('cost')
    routes = solution_data.get('routes', [])

    for route in routes:
        # Check type of route, it is a dict from parse_sol_content
        vehicle_id_raw = route.get('vehicle_id', 0)
        vehicle_id = f"V{vehicle_id_raw}"
        # Ensure path elements are integers and copy to avoid mutation
        path = [int(n) for n in route.get('path', [])]

        if not path:
            continue

        # Skip empty routes (just depot to depot)
        if len(path) == 1 and path[0] == default_depot:
            continue
        if len(path) == 2 and path[0] == default_depot and path[1] == default_depot:
            continue

        # Inject depot if needed at start
        if path[0] != default_depot:
            path.insert(0, default_depot)
        # Inject depot if needed at end
        if path[-1] != default_depot:
            path.append(default_depot)

        # Generate edge segments with sequential ordering per vehicle
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u not in coords:
                continue
            if v not in coords:
                continue

            x1, y1 = coords[u]['x'], coords[u]['y']
            x2, y2 = coords[v]['x'], coords[v]['y']
            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            # Altair rotation is clockwise from North (0 degrees at 12 o'clock)
            # atan2 gives ACW from East.
            # Correct conversion: 90 - degrees(theta)
            angle = 90 - math.degrees(math.atan2(dy, dx))
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Each segment needs two points for line drawing
            # Use global order across all vehicles for proper layering
            seg_order = len(edges)
            edges.append({
                "vehicle_id": vehicle_id,
                "order": seg_order,
                "x": x1, "y": y1,
                "mid_x": mid_x, "mid_y": mid_y,
                "angle": angle,
                "distance": dist
            })
            edges.append({
                "vehicle_id": vehicle_id,
                "order": seg_order + 1,
                "x": x2, "y": y2,
                "mid_x": mid_x, "mid_y": mid_y,
                "angle": angle,
                "distance": dist
            })

    return pd.DataFrame(edges), reported_cost




# --- External APIs ---
def transform_nodes(content: InstanceContent) -> NodesDF:
    """
    External API: Transform InstanceContent into nodes DataFrame.

    Args:
        content: Parsed instance content dictionary

    Returns:
        NodesDF with columns: [id, x, y, demand, type_id, type_label]
    """
    return _build_nodes_df(content)


def transform_edges(
        solution_data: SolutionContent,
        nodes_df: NodesDF,
        default_depot: int = 0
    ) -> tuple[EdgesDF, int | None]:
    """
    External API: Transform SolutionContent into edges DataFrame.

    Args:
        solution_data: Parsed solution dict with 'cost' and 'routes' keys
        nodes_df: DataFrame with node coordinates
        default_depot: Default depot node ID

    Returns:
        - EdgesDF with columns: [vehicle_id, order, x, y, mid_x, mid_y, angle, distance]
        - Reported cost from solution file (or None)
    """
    return _build_edges_df(solution_data, nodes_df, default_depot)


def transform_instance_content(content_str: str) -> NodesDF:
    """
    External API: Parse .sarp content string and transform to nodes DataFrame.
    
    Args:
        content_str: Raw .sarp file content
    
    Returns:
        NodesDF with columns: [id, x, y, demand, type_id, type_label]
    """
    data = parse_sarp_content(content_str)
    return _build_nodes_df(data)


def transform_solution_content(
        content_str: str,
        nodes_df: NodesDF,
        default_depot: int = 0
    ) -> tuple[EdgesDF, int | None]:
    """
    Parse .sol content string and transform to edges DataFrame.
    
    Args:
        content_str: Raw .sol file content
        nodes_df: DataFrame with node coordinates
        default_depot: Default depot node ID
    
    Returns:
        - EdgesDF with columns: [vehicle_id, order, x, y, mid_x, mid_y, angle, distance]
        - Reported cost from solution file (or None)
    """
    solution_data = parse_sol_content(content_str)
    return _build_edges_df(solution_data, nodes_df, default_depot)




# ================ Attempts Transformation ================
def _parse_json_obj(value) -> dict | list | None:
    """
    Parse JSON-like strings (in hyperparams, info, etc.) safely.
    """

    if pd.isna(value):
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError):
        return {}


def _normalize_attempt_df(pre_df: pd.DataFrame) -> AttemptsDF:
    """
    Internal builder: normalize attempts dataframe data types.
    """
    if pre_df.empty:
        return pre_df

    pre_df = pre_df.copy()
    if "attempt_id" in pre_df.columns:
        pre_df["attempt_id"] = pd.to_numeric(pre_df["attempt_id"], errors="coerce")
    if "timestamp" in pre_df.columns:
        pre_df["timestamp"] = pd.to_datetime(pre_df["timestamp"], errors="coerce")
    if "time_limit" in pre_df.columns:
        pre_df["time_limit"] = pd.to_numeric(pre_df["time_limit"], errors="coerce")
    if "elapsed_time" in pre_df.columns:
        pre_df["elapsed_time"] = pd.to_numeric(pre_df["elapsed_time"], errors="coerce")
    if "cost" in pre_df.columns:
        pre_df["cost"] = pd.to_numeric(pre_df["cost"], errors="coerce")
    if "seed" in pre_df.columns:
        pre_df["seed"] = pd.to_numeric(pre_df["seed"], errors="coerce")
    if "hyperparams" in pre_df.columns:
        pre_df["hyperparams"] = pre_df["hyperparams"].apply(_parse_json_obj)
    if "info" in pre_df.columns:
        pre_df["info"] = pre_df["info"].apply(_parse_json_obj)
    if "status" in pre_df.columns:
        pre_df['status'] = pre_df['status'].astype(str)
    return pre_df


def _normalize_scoreboard_df(pre_df: pd.DataFrame) -> ScoreboardDF:
    """
    Internal builder: normalize scoreboard dataframe data types.
    """
    if pre_df.empty:
        return pre_df

    work = pre_df.copy()
    if "best_timestamp" in pre_df.columns:
        work["best_timestamp"] = pd.to_datetime(work["best_timestamp"], errors="coerce")
    numeric_cols = [
        "best_cost", "best_attempt_id", "best_time_taken", "num_attempts", 
        "successful_attempts", "cost_improvement", "percentage_improvement"
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    for col in ("best_solver_args", "best_solver_hyperparams"):
        if col in work.columns:
            work[col] = work[col].apply(_parse_json_obj)

    return work


def transform_attempts_df(file_path: str) -> AttemptsDF:
    """
    External API: Read attempts CSV and return normalized AttemptsDF.
    
    Args:
        file_path: Path to the attempts CSV file
    
    Returns:
        Normalized AttemptsDF with proper data types
    """
    raw_df = extract_csv_dataframe(file_path, ATTEMPT_DF_COLUMNS)
    return _normalize_attempt_df(raw_df)


def transform_scoreboard_df(file_path: str) -> ScoreboardDF:
    """
    External API: Read scoreboard CSV and return normalized ScoreboardDF.
    
    Args:
        file_path: Path to the scoreboard CSV file
    
    Returns:
        Normalized ScoreboardDF with proper data types
    """
    raw_df = extract_csv_dataframe(file_path, SCOREBOARD_DF_COLUMNS)
    return _normalize_scoreboard_df(raw_df)




# ================ Performance Tracking Transformation ================
def _empty_df(columns: list[str]) -> pd.DataFrame:
    """Helper: Return an empty dataframe with the given columns."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _build_top_best_attempts(att_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Internal builder: Return dataframe of the best -valid- `n` attempts 
    across **each** instances in `one` dataset.
    """
    columns = [
        "instance", "attempt_id", "timestamp", "solver",
        "status", "cost", "elapsed_time", "note",
    ]

    if att_df.empty:
        return _empty_df(columns)
    # Reject only 'error' status, include 'done', 'time_limit', etc.
    work: pd.DataFrame = att_df.loc[att_df['status'].str.lower() != "error"].copy()
    if work.empty:
        return _empty_df(columns)

    best_df = (
        work
        .sort_values(
            ["instance", "cost", "timestamp"],
            ascending=[True, True, True],
            kind="mergesort"
        )
        .groupby("instance")
        .head(n)
        .reset_index(drop=True)
        .reindex(columns=columns)
        .copy()
    )
    return best_df


def _build_top_recent_attempts(inst: str, att_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Internal builder: Return dataframe of the most recent -valid- `n` attempts 
    across **one** instance.
    """
    columns: list[str] = [
        "instance", "attempt_id", "timestamp", "solver",
        "status", "cost", "elapsed_time", "note",
    ]

    if att_df.empty:
        return _empty_df(columns)
    mask = (
        (att_df["instance"] == inst)
        # Reject only 'error' status, include 'done', 'time_limit', etc.
        & (att_df['status'].astype(str).str.lower() != "error")
    )
    work: pd.DataFrame = att_df.loc[mask, columns].copy()
    if work.empty:
        return _empty_df(columns)

    recent_df = (
        work
        .sort_values(["timestamp"], ascending=[False], kind="mergesort")
        .groupby("instance")
        .head(n)
        .reset_index(drop=True)
        .reindex(columns=columns)
        .copy()
    )
    return recent_df


def _build_recent_vs_best_attempts(
        inst: str,
        recent_df: pd.DataFrame,
        score_df: pd.DataFrame,
        n: int = 3
    ) -> pd.DataFrame:
    """
    Internal builder: Build dataframe comparing the most recent `n` attempts 
    versus the best attempt across **one** instance.
    """
    precolumns = pd.Index([
        "attempt_id", "timestamp", "solver", "status", "cost", "note",
    ])
    columns = pd.Index([
        "attempt_id", "timestamp", "solver", "status", "cost", "cost_gap", "pct_gap", "note",
    ])

    best_row_df: pd.DataFrame = score_df.loc[score_df["instance"] == inst]
    if best_row_df.empty:
        return _empty_df(list(columns))
    best_row = best_row_df.iloc[0]

    best_entry = pd.DataFrame([{
        "attempt_id": best_row.get("best_attempt_id", pd.NA),
        "timestamp": best_row.get("best_timestamp", pd.NaT),
        "solver": best_row.get("best_solver", pd.NA),
        "status": "done",
        "cost": best_row.get("best_cost", pd.NA),
        "note": best_row.get("notes", None),
    }])

    best_cost_raw = best_row.get("best_cost")
    best_cost_val: int | None = int(best_cost_raw) if pd.notna(best_cost_raw) else None

    recent_attempts: pd.DataFrame = (
        recent_df
        .sort_values("timestamp", ascending=False, kind="mergesort")
        .head(n)
        .copy()
    )

    vs_df = pd.concat(
        [
            best_entry.reindex(columns=precolumns),
            recent_attempts.reindex(columns=precolumns),
        ],
        ignore_index=True,
    )

    vs_df["cost"] = pd.to_numeric(vs_df["cost"], errors="coerce")
    if best_cost_val is not None and best_cost_val != 0:
        vs_df["cost_gap"] = vs_df["cost"] - best_cost_val
        vs_df["pct_gap"] = (vs_df["cost_gap"] / best_cost_val * 100).round(2)
    else:
        vs_df["cost_gap"] = pd.NA
        vs_df["pct_gap"] = pd.NA

    return vs_df.reindex(columns=columns)


def _build_attempts_summary(att_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Internal builder: Build summary dataframe of the most recent `n` attempts 
    across **all** instances and solvers.
    """
    columns = pd.Index([
        "instance", "attempt_id", "timestamp", "solver",
        "status", "cost", "elapsed_time", "note"
    ])

    if att_df.empty:
        return pd.DataFrame(columns=columns)

    recent_attempts = (
        att_df
        .reindex(columns=columns)
        .sort_values("timestamp", ascending=False)
        .head(n)
        .reset_index(drop=True)
        .copy()
    )
    return recent_attempts


def _build_solvers_summary(att_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal builder: Build dataframe summarizing all solver's performances 
    across **all instances.
    """
    columns = [
        "solver", "num_attempts", "successes", "success_rate",
        "last_instance", "last_attempt_id", "last_status", "last_timestamp",
        "last_cost", "last_elapsed_time", "last_note",
    ]

    if att_df.empty:
        return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
    work: pd.DataFrame = att_df.copy().reset_index(drop=True)

    stats = (
        work
        .groupby("solver", dropna=False)['status']
        .agg(
            num_attempts="size",
            successes=lambda s: (s.str.lower() != "error").sum(),
            success_rate=lambda s: (s.str.lower() != "error").mean()
        )
        .reset_index()
    )

    last_att_by_solver: pd.DataFrame = (
        work
        .sort_values(["solver", "timestamp"], ascending=[True, True], kind="mergesort")
        .groupby("solver", dropna=False)
        .last()
        .reset_index()
        .loc[:, [
            "solver", "instance", "attempt_id", "status", "timestamp",
            "cost", "elapsed_time", "note"
        ]]
        .rename(columns={
            "instance": "last_instance",
            "attempt_id": "last_attempt_id",
            "status": "last_status",
            "timestamp": "last_timestamp",
            "cost": "last_cost",
            "elapsed_time": "last_elapsed_time",
            "note": "last_note",
        })
    )

    summary = stats.merge(
        right=last_att_by_solver, on="solver", how="left"
    ).reindex(columns=columns).copy()

    return summary


def _build_solver_leaderboard(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal builder: Compute solver leadership counts from the scoreboard dataframe.
    """
    columns = ["solver", "num_instances", "instances"]
    work = score_df.copy().dropna(subset=["best_solver", "instance"])
    if work.empty:
        return _empty_df(columns)

    grouped = work.groupby("best_solver")
    leaderboard = (
        grouped["instance"]
        .agg(
            num_instances="nunique",
            instances=lambda s: sorted(s.unique()),
        )
        .reset_index()
        .rename(columns={"best_solver": "solver"})
    )

    return leaderboard.sort_values(
        by=["num_instances", "solver"], ascending=[False, True]
    ).reset_index(drop=True)


def _build_solver_config(att_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Internal builder: Return configuration tracking dataframe for solvers 
    in recent attempts.
    """
    columns = [
        "solver", "instance", "attempt_id", "timestamp", "status",
        "cost", "seed", "time_limit", "hyperparams", "note",
    ]

    work = (
        att_df
        .copy()
        .reindex(columns=columns)
        .dropna(subset=["timestamp", "solver"])
        .sort_values(["timestamp", "solver"], ascending=[True, False])
        .groupby("solver", dropna=False)
        .head(limit)
        .reset_index(drop=True)
    )

    return work




# --- External APIs ---
def transform_top_best_attempts(att_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return dataframe of the best successful `n` attempts per instance."""
    return _build_top_best_attempts(att_df, n)


def transform_top_recent_attempts(inst: str, att_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return dataframe of the most recent successful `n` attempts for one instance."""
    return _build_top_recent_attempts(inst, att_df, n)


def transform_recent_vs_best(
        inst: str,
        recent_df: pd.DataFrame,
        score_df: pd.DataFrame,
        n: int = 3
    ) -> pd.DataFrame:
    """Compare most recent `n` attempts versus the best attempt."""
    return _build_recent_vs_best_attempts(inst, recent_df, score_df, n)


def transform_attempts_summary(att_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Build summary dataframe of the most recent `n` attempts."""
    return _build_attempts_summary(att_df, n)


def transform_solvers_summary(att_df: pd.DataFrame) -> pd.DataFrame:
    """Build dataframe summarizing all solver's performances."""
    return _build_solvers_summary(att_df)


def transform_solver_leaderboard(score_df: pd.DataFrame) -> pd.DataFrame:
    """Compute solver leadership counts from the scoreboard."""
    return _build_solver_leaderboard(score_df)


def transform_solver_config(att_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Return configuration tracking dataframe for solvers."""
    return _build_solver_config(att_df, limit)
