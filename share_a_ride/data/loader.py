"""
This module processes data generated during experiments for research, 
error analysis, insight exploration, and dashboard reporting.

It provides functionalities to:
- Load raw data from local or remote sources (e.g., databases, APIs, CSV files).
- Clean, transform, and standardize data for analysis.
- Cache or export intermediate datasets for faster dashboard rendering.
- Ensure data consistency and integrity across processing stages.

It solves the problem of ensuring a smooth workflow of loading and benchmarking 
for this Share-a-Ride optimization project.

It serves for 3 main business domains:
- Navigate and select datasets throughout the experiment via a catalog.
- Provide per-solver and per-instance performance tracking and summaries.
- Tracking solver configurations over time.
"""
import os
import json
import ast
import math
from functools import lru_cache

import pandas as pd

from share_a_ride.data.router import path_router
from share_a_ride.data.classes import Dataset




# ================ Utility functions ================
def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _read_pd_from_csv_file(path: str, columns: list[str]) -> pd.DataFrame:
    """ 
    CSV reader with fallback to empty dataframe.
    Receive the **absolute** path and the columns should be read.
    Return a dataframe with the given columns if the file does not exist or is empty.
    """
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return _empty_df(columns)

    df: pd.DataFrame = pd.read_csv(path)
    return df.reindex(columns=columns)


def _parse_json_like(value) -> dict | list | None:
    """
    Parse JSON-like strings (in hyperparams, info, etc.) safely without raising.

    Returns dict | list | None and tolerates common formats:
    - Proper JSON
    - Python literals (via ast.literal_eval)
    - Single-quoted JSON and True/False/None variants

    On failure, returns an empty dict to avoid breaking downstream code.
    """
    # Treat NaN/NA as None
    if pd.isna(value):
        return None

    # Pass-through already structured values
    if isinstance(value, (dict, list)):
        return value

    # Normalize to string for parsing
    text = str(value).strip()
    if not text:
        return {}

    # Try standard JSON first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback to Python literal (ast.literal_eval) for single-quoted strings
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError):
        # Final fallback: return empty dict if both fail
        return {}


def _coerce_optional_int(value: object) -> int | None:
    """Best-effort coercion of a dynamic value to int for strict type checking.

    Accepts ints, finite floats, and numeric strings (int or float-like),
    returns None for None/NaN/empty/invalid values. Avoids casting Any and
    avoids pandas to_numeric to keep the type narrow for type checkers.
    """
    if value is None:
        return None
    # avoid treating booleans as integers
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            # allow float-like strings ("12.0") but coerce to int
            return int(float(text))
        except ValueError:
            return None
    # unsupported type
    return None




# ================ Domain 1: Navigation catalog ================
def _build_dataset_catalog() -> pd.DataFrame:
    """
    Build catalog dataframe mapping dataset resources down to instance level.
    Returns a dataframe with the following columns:
    - purpose: purpose of the dataset (e.g., "test", "train", etc.)
    - dataset: name of the dataset
    - instance: instance identifier (NaN when none available)
    """

    columns = pd.Index(["purpose", "dataset", "instance"])
    rows: list[dict[str, str]] = []

    for ds_name, purpose in sorted(
        [(d.value.name, d.value.purpose.value) for d in Dataset],
        key=lambda item: (item[1], item[0])
    ):
        # Resolve the dataset root folder once
        dataset_root = path_router(ds_name, "readall")

        # Collect unique instance names (.sarp) under this dataset root
        instances_set: set[str] = set()
        if isinstance(dataset_root, str) and os.path.isdir(dataset_root):
            for _root, _dirs, files in os.walk(dataset_root, topdown=True, followlinks=False):
                for file in files:
                    if file.endswith(".sarp"):
                        instances_set.add(file[:-5])  # strip extension

        for inst in sorted(instances_set):
            rows.append({
                "purpose": purpose,
                "dataset": ds_name,
                "instance": inst,
            })

    if not rows:
        return _empty_df(list(columns))

    catalog = (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["purpose", "dataset", "instance"], kind="mergesort")
        .reset_index(drop=True)
    )
    return catalog




# =============== Domain 2: Processing experiments data ================
def normalize_attempts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types in attempts dataframe.
    """
    if df.empty:
        return df

    df = df.copy()
    if "attempt_id" in df.columns:
        df["attempt_id"] = pd.to_numeric(df["attempt_id"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "time_limit" in df.columns:
        df["time_limit"] = pd.to_numeric(df["time_limit"], errors="coerce")
    if "elapsed_time" in df.columns:
        df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="coerce")
    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    if "hyperparams" in df.columns:
        df["hyperparams"] = df["hyperparams"].apply(_parse_json_like)
    if "info" in df.columns:
        df["info"] = df["info"].apply(_parse_json_like)
    if "status" in df.columns:
        df['status'] = df['status'].astype(str)
    return df


def normalize_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types in scoreboard dataframe.
    Return the normalized dataframe.
    """
    if df.empty:
        return df

    df = df.copy()
    if "best_timestamp" in df.columns:
        df["best_timestamp"] = pd.to_datetime(df["best_timestamp"], errors="coerce")

    numeric_cols = [
        "best_cost", "best_attempt_id", "best_time_taken", "num_attempts", 
        "successful_attempts", "cost_improvement", "percentage_improvement"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("best_solver_args", "best_solver_hyperparams"):
        if col in df.columns:
            df[col] = df[col].apply(_parse_json_like)

    return df




# ================ Domain 2: Performance Tracking ================
def _build_top_best_attempts(att_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Return dataframe of the best -successful- `n` attempts across 
    **each** instances in `one` dataset.
    The att_df is taken from normalize_attempts.
    n is the number of best attempts to return.
    """

    columns = [
        "instance",
        "attempt_id",
        "timestamp",
        "solver",
        "status",
        "cost",
        "elapsed_time",
        "note",
    ]

    # Retrieve successful attempts
    if att_df.empty:
        return _empty_df(columns)
    work: pd.DataFrame = att_df.loc[att_df['status'].str.lower() == "done"].copy()
    if work.empty:
        return _empty_df(columns)

    # Retrieve the best n attempts per instance in the dataset
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
    Return dataframe of the most recent -successful- `n` attempts across **one** instance.
    The att_df is taken from normalize_attempts.
    n is the number of recent successful attempts to return.
    """

    columns: list[str] = [
        "instance",
        "attempt_id",
        "timestamp",
        "solver",
        "status",
        "cost",
        "elapsed_time",
        "note",
    ]

    # Retrieve successful attempts
    if att_df.empty:
        return _empty_df(columns)
    # Filter by instance and successful status (case-insensitive)
    mask = (
        (att_df["instance"] == inst)
        & (att_df['status'].astype(str).str.lower() == "done")
    )
    work: pd.DataFrame = att_df.loc[mask, columns].copy()
    if work.empty:
        return _empty_df(columns)

    # Retrieve most recent `n` attempts
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
    Build dataframe comparing the most recent `n` attempts versus the best attempt 
    across **one** instance, with the recent attempts taken from the recent attempts
    record retrieved above, and the best attempt taken from scoreboard dataframe.

    The two additional columns are:
    - cost_gap: difference between latest attempt cost and best attempt cost
    - pct_gap: percentage gap between latest attempt cost and best attempt cost

    If the entry is the best attempt, these columns are zero.
    """
    precolumns = pd.Index([
        "attempt_id",
        "timestamp",
        "solver",
        "status",
        "cost",
        "note",
    ])
    columns  = pd.Index([
        "attempt_id",
        "timestamp",
        "solver",
        "status",
        "cost",
        "cost_gap",
        "pct_gap",
        "note",
    ])

    # Retrieve best attempt row for this instance from the scoreboard
    best_row_df: pd.DataFrame = score_df.loc[score_df["instance"] == inst]
    if best_row_df.empty:
        return _empty_df(list(columns))
    best_row = best_row_df.iloc[0]

    # Build a normalized one-row dataframe representing the best attempt
    best_entry = pd.DataFrame([
        {
            "attempt_id": best_row.get("best_attempt_id", pd.NA),
            "timestamp": best_row.get("best_timestamp", pd.NaT),
            "solver": best_row.get("best_solver", pd.NA),
            # Best entry implies success
            "status": "done",
            "cost": best_row.get("best_cost", pd.NA),
            # Scoreboard uses 'notes' (optional); map to 'note' here
            "note": best_row.get("notes", None),
        }
    ])

    # Extract best cost as Optional[int] (strict type-safe)
    best_cost_val: int | None = _coerce_optional_int(best_row.get("best_cost"))

    # Retrieve most recent `n` attempts
    recent_attempts: pd.DataFrame = (
        recent_df
        .sort_values("timestamp", ascending=False, kind="mergesort")
        .head(n)
        .copy()
    )

    # Combine best attempt and recent attempts
    vs_df = pd.concat(
        [
            best_entry.reindex(columns=precolumns),
            recent_attempts.reindex(columns=precolumns),
        ],
        ignore_index=True,
    )

    # Compute cost gaps (handle NaNs gracefully)
    vs_df["cost"] = pd.to_numeric(vs_df["cost"], errors="coerce")
    if best_cost_val is not None and best_cost_val != 0:
        vs_df["cost_gap"] = vs_df["cost"] - best_cost_val
        vs_df["pct_gap"] = (vs_df["cost_gap"] / best_cost_val * 100).round(2)
    else:
        vs_df["cost_gap"] = pd.NA
        vs_df["pct_gap"] = pd.NA

    return vs_df.reindex(columns=columns)


def _build_attempts_summary(
        att_df: pd.DataFrame,
        n: int = 10
    ) -> pd.DataFrame:
    """
    Build summary dataframe of the most recent `n` attempts 
    across **all** instances and solvers.
    """

    columns  = pd.Index([
        "instance",
        "attempt_id", 
        "timestamp", 
        "solver",
        "status", 
        "cost", 
        "elapsed_time", 
        "note"
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
    Build dataframe summarizing all solver's performances across **all instances.
    Returns a dataframe with the following columns:
    - solver: name of the solver
    - num_attempts: total number of attempts made by the solver
    - successes: number of successful attempts
    - success_rate: ratio of successful attempts to total attempts
    - last_instance: the instance of the most recent attempt
    - last_attempt_id: the attempt_id of the most recent attempt
    - last_status: the status of the most recent attempt
    - last_timestamp: the timestamp of the most recent attempt
    - last_cost: the cost of the most recent attempt
    - last_elapsed_time: the elapsed_time of the most recent attempt
    - last_note: the note of the most recent attempt
    """

    columns = [
        "solver",
        "num_attempts",
        "successes",
        "success_rate",
        "last_instance",
        "last_attempt_id",
        "last_status",
        "last_timestamp",
        "last_cost",
        "last_elapsed_time",
        "last_note",
    ]

    # Retrieve working copy
    if att_df.empty:
        return pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
    work: pd.DataFrame = att_df.copy().reset_index(drop=True)

    # Retrieve attempt stats
    stats = (
        work
        .groupby("solver", dropna=False)['status']
        .agg(
            num_attempts="size",
            successes=lambda s: (s.str.lower() == "done").sum(),
            success_rate=lambda s: (s.str.lower() == "done").mean()
        )
        .reset_index()
    )

    # Retrieve last attempt info, by finding the last attempt per solver by timestamp,
    # assuming that the last attempt per solver is in the end of att_df, group by solver.
    # If the files were handled correctly, this should be true.

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

    # Merge aggregates with last-attempt info
    summary = stats.merge(
        right=last_att_by_solver, on="solver", how="left"
    ).reindex(columns=columns).copy()


    return summary


def _build_solver_leaderboard(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute solver leadership counts of the current dataset from the scoreboard dataframe.
    
    Return a dataframe with the following columns:
    - solver: name of the solver
    - num_instances: number of instances where the solver is the best
    - instances: list of instances where the solver is the best
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




# =============== Domain 3: Solver Configuration Tracking ================
def _build_solver_config(att_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """
    Return configuration tracking dataframe for solvers in recent attempts.
    This includes seed, time limit and hyperparameters.
    This is useful for tracking hyperparameter changes over time in the short term.
    """

    columns = [
        "solver",
        "instance",
        "attempt_id",
        "timestamp",
        "status",
        "cost",
        "seed",
        "time_limit",
        "hyperparams",
        "note",
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




# ================ Main data loader class ================
class DataLoader:
    """
    API surface for experimental data access. 
    The workflow includes:
    - Loads raw CSVs from disk (local or remote) via path_router.
    - Normalizes raw CSVs once (per dataset) via an internal cache.
    - Processes data into useful dataframes for dashboard consumption.

    The main logic it provides includes:
    - Dataset catalog loading and navigation.
    - Dataset-level data bundle loading for dashboard.
    - Instance-scoped data loading for charts that require selection.
    """

    catalog_cache: pd.DataFrame | None = None
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


    # ---------- Internal cached loaders ----------
    @staticmethod
    @lru_cache(maxsize=16)
    def _load_normalized_frames(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        attempts_path = path_router(dataset, "record")
        scoreboard_path = path_router(dataset, "summarize")

        attempts_df = normalize_attempts(
            _read_pd_from_csv_file(
                attempts_path, DataLoader.ATTEMPT_DF_COLUMNS
            )
        )
        scoreboard_df = normalize_scoreboard(
            _read_pd_from_csv_file(
                scoreboard_path, DataLoader.SCOREBOARD_DF_COLUMNS
            )
        )
        return attempts_df, scoreboard_df


    # --------------- Catalog navigation helpers ---------------
    @classmethod
    def load_catalog_frame(cls, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load and return catalog dataframe:
        columns: ['purpose', 'dataset', 'instance']
        """
        if force_refresh or cls.catalog_cache is None:
            cls.catalog_cache = _build_dataset_catalog()
        return cls.catalog_cache.copy()


    @classmethod
    def list_purposes(cls) -> list[str]:
        """
        Return list of unique purposes in the catalog.
        """
        catalog = cls.load_catalog_frame()
        return catalog["purpose"].dropna().unique().tolist()


    @classmethod
    def list_datasets(cls, purpose: str) -> list[str]:
        """
        Return sorted list of datasets for a given purpose.
        """
        catalog = cls.load_catalog_frame()
        return (
            catalog.loc[catalog["purpose"] == purpose, "dataset"]
            .dropna()
            .unique()
            .tolist()
        )


    @classmethod
    def list_instances(cls, dataset: str) -> list[str]:
        """
        Return sorted list of instances for a given dataset.
        """
        catalog = cls.load_catalog_frame()
        instances = (
            catalog.loc[catalog["dataset"] == dataset, "instance"]
            .dropna()
            .unique()
            .tolist()
        )
        instances.sort()
        return instances


    @classmethod
    def get_catalog_tree(cls, force_refresh: bool = False) -> dict[str, dict[str, list[str]]]:
        """
        Return a nested mapping {purpose: {dataset: [instances...]}}.
        This is useful for building a collapsible catalog view.
        """
        df = cls.load_catalog_frame(force_refresh=force_refresh)
        tree: dict[str, dict[str, list[str]]] = {}
        for purpose in df["purpose"].dropna().unique():
            sub: pd.DataFrame = df.loc[df["purpose"] == purpose]
            datasets = sub["dataset"].dropna().unique()
            tree[purpose] = {}
            for ds in datasets:
                insts = (
                    sub.loc[sub["dataset"] == ds, "instance"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                insts.sort()
                tree[purpose][ds] = insts

        return tree



    # ---------- Dataset-level bundle for dashboard ----------
    @classmethod
    def load_dataset_bundle(cls, dataset: str) -> dict[str, pd.DataFrame]:
        """
        Return a minimal bundle of dfs needed by the dashboard for one dataset:
        - attempts: processed attempts record taken from the attempts CSV
        - scoreboard: processed scoreboard record taken from the summarize CSV
        - recent_attempts: _build_attempts_summary(attempts, n=10)
        - solver_summary: _build_solvers_summary(attempts)
        - best_attempts: _build_top_best_attempts(attempts, n=3)
        - solver_leaderboard: _build_solver_leaderboard(scoreboard)
        - solver_recent_hyperparams: _build_solver_config(attempts, limit=3)

        This is useful for building the main dashboard view of the dataset.
        Further instance-scoped charts can be built via instance-scoped APIs.
        """
        try:
            attempts_df, scoreboard_df = cls._load_normalized_frames(dataset)
        except KeyError as exc:
            raise ValueError(f"Unknown dataset '{dataset}'") from exc

        attempts_df = attempts_df.copy()
        scoreboard_df = scoreboard_df.copy()

        # Build only with existing helpers (no undefined calls)
        best_attempts_df = _build_top_best_attempts(attempts_df, n=3)
        recent_attempt_df = _build_attempts_summary(attempts_df, n=10)
        solver_summary_df = _build_solvers_summary(attempts_df)
        solver_leaderboard_df = _build_solver_leaderboard(scoreboard_df)
        solver_recent_hparams_df = _build_solver_config(attempts_df, limit=3)

        return {
            "attempts": attempts_df,
            "scoreboard": scoreboard_df,
            "best_attempts": best_attempts_df,
            "recent_attempts": recent_attempt_df,
            "solver_summary": solver_summary_df,
            "solver_leaderboard": solver_leaderboard_df,
            "solver_recent_hyperparams": solver_recent_hparams_df,
        }



    # ---------- Instance-scoped APIs for charts that require selection ----------
    @classmethod
    def load_instance_recent(
            cls, dataset: str, instance: str, n: int = 3
        ) -> pd.DataFrame:
        """
        Return most recent successful attempts for a given instance:
        _build_top_recent_attempts(instance, attempts, n)
        """
        attempts_df, _ = cls._load_normalized_frames(dataset)
        return _build_top_recent_attempts(instance, attempts_df, n=n)


    @classmethod
    def load_recent_vs_best(
            cls, dataset: str, instance: str, n: int = 3
        ) -> pd.DataFrame:
        """
        Return recent vs best attempts for a given instance:
        """

        attempts_df, scoreboard_df = cls._load_normalized_frames(dataset)
        recent_df = _build_top_recent_attempts(instance, attempts_df, n=n)
        return _build_recent_vs_best_attempts(instance, recent_df, scoreboard_df, n=n)



    @classmethod
    def load_instance_bundle(
            cls, dataset: str, instance: str, n: int = 3
        ) -> dict[str, pd.DataFrame]:
        """
        Return a minimal bundle of dfs needed by instance-scoped charts:
        - instance_recent: _build_top_recent_attempts(instance, attempts, n)
        - instance_recent_vs_best: _build_recent_vs_best_attempts(
            instance, instance_recent, scoreboard, n)
        """
        attempts_df, scoreboard_df = cls._load_normalized_frames(dataset)
        instance_recent = _build_top_recent_attempts(instance, attempts_df, n=n)
        instance_recent_vs_best = _build_recent_vs_best_attempts(
            instance, instance_recent, scoreboard_df, n=n
        )
        return {
            "instance_recent": instance_recent,
            "instance_recent_vs_best": instance_recent_vs_best,
        }



    # ---------- Misc convenience ----------
    @classmethod
    def load_solver_leaderboard(cls, dataset: str) -> pd.DataFrame:
        """Convenience: leaderboard only."""
        _, scoreboard_df = cls._load_normalized_frames(dataset)
        return _build_solver_leaderboard(scoreboard_df)


    @classmethod
    def load_solver_recent_config(cls, dataset: str, limit: int = 3) -> pd.DataFrame:
        """Convenience: recent solver configs only."""
        attempts_df, _ = cls._load_normalized_frames(dataset)
        return _build_solver_config(attempts_df, limit=limit)


    @classmethod
    def invalidate_caches(cls) -> None:
        """Clear cached catalog and normalized frames (useful for Refresh)."""
        cls.catalog_cache = None
        try:
            cls._load_normalized_frames.cache_clear()
        except AttributeError:
            pass


    # ---------- Lightweight helpers for debugging/perf ----------
    @classmethod
    def list_all_datasets(cls) -> list[str]:
        """
        Fast dataset listing without crawling instances.
        Returns sorted keys from Dataset enum.
        """
        return sorted([d.value.name for d in Dataset])




if __name__ == "__main__":
    pass
    # Debug runner
    # Usage:
    #   python share_a_ride/data/loader.py                 # quick run on first dataset
    #   python share_a_ride/data/loader.py DATASET         # run on specified dataset
    #   python share_a_ride/data/loader.py --all           # iterate all datasets (summary)

    # def _print_df(name: str, df: pd.DataFrame | None, n: int = 5) -> None:
    #     try:
    #         rows = 0 if df is None else len(df)
    #         cols = [] if df is None else list(df.columns)
    #         print(
    #             f"\n[{name}] rows={rows} cols={len(cols)}"
    #             f" -> {cols[:8]}{'...' if len(cols)>8 else ''}"
    #         )
    #         if df is not None and not df.empty:
    #             print(df.head(n).to_string(index=False))

    #     except Exception as exc:
    #         print(f"[{name}] ERROR: {exc}")

    # def debug_catalog() -> None:
    #     """
    #     Debug printout of the catalog and its stats.
    #     """
    #     print("\n== Catalog ==")
    #     cat = DataLoader.load_catalog_frame(force_refresh=True)
    #     # Show dataset distribution and per-purpose coverage to avoid confusion
    #     try:
    #         if isinstance(cat, pd.DataFrame) and not cat.empty:
    #             ds_counts = cat["dataset"].value_counts().sort_index()
    #             print("dataset row counts:")
    #             print(ds_counts.to_string())

    #             purpose_counts = (
    #                 cat.groupby(["purpose"], dropna=False)["dataset"]
    #                 .nunique()
    #                 .sort_index()
    #             )
    #             print("datasets per purpose:")
    #             print(purpose_counts.to_string())

    #             head_by_ds = (
    #                 cat.sort_values(["purpose", "dataset", "instance"], kind="mergesort")
    #                    .groupby("dataset", as_index=False)
    #                    .head(1)
    #             )
    #             print("\nfirst instance per dataset (up to 10):")
    #             print(head_by_ds.head(10).to_string(index=False))
    #     except Exception as exc:
    #         print("catalog stats ERROR:", exc)
    #     tree = DataLoader.get_catalog_tree(force_refresh=True)
    #     print(f"catalog purposes={len(tree)} datasets={sum(len(v) for v in tree.values())}")


    # def debug_dataset(ds_name: str) -> None:
    #     """
    #     Debug printout of dataset-level and instance-level data.
    #     """
    #     print(f"\n== Dataset: {ds_name} ==")
    #     try:
    #         bundle = DataLoader.load_dataset_bundle(ds_name)
    #     except Exception as exc:
    #         print("load_dataset_bundle ERROR:", exc)
    #         return
    #     for key, df in bundle.items():
    #         _print_df(f"bundle.{key}", df, n=5)

    #     try:
    #         cfg_full = DataLoader.load_solver_recent_config(ds_name, limit=20)
    #         _print_df("cfg_full", cfg_full, n=5)
    #     except Exception as exc:
    #         print("cfg_full ERROR:", exc)

    #     try:
    #         lb = DataLoader.load_solver_leaderboard(ds_name)
    #         _print_df("leaderboard", lb, n=5)
    #     except Exception as exc:
    #         print("leaderboard ERROR:", exc)

    #     # Instance-level checks (pick the first instance if any)
    #     instances = []
    #     try:
    #         instances = DataLoader.list_instances(ds_name)
    #     except Exception:
    #         instances = []
    #     inst = instances[0] if instances else None
    #     print("instances count:", len(instances), "sample:", inst)
    #     if inst:
    #         try:
    #             r = DataLoader.load_instance_recent(ds_name, inst, n=3)
    #             _print_df("instance_recent", r, n=3)
    #         except Exception as exc:
    #             print("instance_recent ERROR:", exc)
    #         try:
    #             rvb = DataLoader.load_recent_vs_best(ds_name, inst, n=3)
    #             _print_df("recent_vs_best", rvb, n=3)
    #         except Exception as exc:
    #             print("recent_vs_best ERROR:", exc)
    #         try:
    #             ib = DataLoader.load_instance_bundle(ds_name, inst, n=3)
    #             for k, df in ib.items():
    #                 _print_df(f"instance_bundle.{k}", df, n=3)
    #         except Exception as exc:
    #             print("instance_bundle ERROR:", exc)


    # args = sys.argv[1:]
    # if args and args[0] == "--all":
    #     print("Iterating all datasets (summary mode)...")
    #     for ds_name_ in DataLoader.list_all_datasets():
    #         try:
    #             # Only print brief summaries to keep output readable
    #             print(f"\n== {ds_name_} ==")
    #             bundle = DataLoader.load_dataset_bundle(ds_name_)
    #             print(
    #                 "attempts rows:", len(bundle.get("attempts", pd.DataFrame())),
    #                 "scoreboard rows:", len(bundle.get("scoreboard", pd.DataFrame()))
    #             )
    #         except Exception as exc:
    #             print(f"{ds_name_} ERROR:", exc)
    #     sys.exit(0)

    # selected_dataset = args[0] if args else DataLoader.list_all_datasets()[0]
    # if not selected_dataset:
    #     print("No datasets configured. Check Dataset enum in classes.")
    #     sys.exit(1)

    # debug_catalog()
    # debug_dataset(selected_dataset)
