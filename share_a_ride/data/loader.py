"""
Data loader module for Share-a-Ride experimental data.
This module provides an API surface for loading and processing experimental data
for dashboard and visualizer components.

It serves as the Load layer in the ETL pipeline, building on top of:
- Extractor: for raw data extraction from files
- Transformer: for data normalization and processing into useful dataframes
"""

from functools import lru_cache
from typing import TypedDict, Tuple, Any

from numpy import ndarray
import pandas as pd

from share_a_ride.data.router import path_router
from share_a_ride.data.classes import Dataset
from share_a_ride.data.extractor import extract_sarp_content, extract_sol_content
from share_a_ride.data.transformer import (
    # Catalog transformation
    transform_catalog,

    # Attempts transformation
    transform_attempts_df,
    transform_scoreboard_df,

    # Performance tracking transformation
    transform_top_best_attempts,
    transform_top_recent_attempts,
    transform_recent_vs_best,
    transform_attempts_summary,
    transform_solvers_summary,
    transform_solver_leaderboard,
    transform_solver_config,

    # Visualizer transformation
    transform_nodes,
    transform_edges,
    transform_instance_content,
    transform_solution_content,

    # Type aliases
    AttemptsDF,
    ScoreboardDF,
    NodesDF,
    EdgesDF,
    CatalogDF,
)




# ================ Bundle Definitions ================
DatasetBundle = Tuple[AttemptsDF, ScoreboardDF]


class InstanceBundle(TypedDict):
    """Type definition for instance-scoped data bundle."""
    instance_recent: pd.DataFrame
    instance_recent_vs_best: pd.DataFrame


class DashboardBundle(TypedDict):
    """Type definition for dataset dashboard data bundle."""
    attempts: pd.DataFrame
    scoreboard: pd.DataFrame
    best_attempts: pd.DataFrame
    recent_attempts: pd.DataFrame
    solver_summary: pd.DataFrame
    solver_leaderboard: pd.DataFrame
    solver_recent_hyperparams: pd.DataFrame


class VisualizerBundle(TypedDict):
    """Type definition for visualizer data bundle."""
    nodes_df: NodesDF
    edges_df: EdgesDF
    reported_cost: int | None




# ================ Constants ================
# Path to the main solution warehouse file
SOLUTION_WAREHOUSE_PATH = "share_a_ride/data/route.sol"




# ================ Utility functions ================
def _empty_df(columns: list[str]) -> pd.DataFrame:
    """
    Return an empty dataframe with the given columns.
    """
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})




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




    # ================ Miscellaneous ================
    @classmethod
    def invalidate_caches(cls) -> None:
        """Clear cached catalog and normalized frames (useful for Refresh)."""
        cls.catalog_cache = None
        try:
            cls.load_dataset_bundle.cache_clear()
        except AttributeError:
            pass


    @classmethod
    def list_purposes(cls) -> list[str]:
        """
        Return list of unique purposes in the catalog.
        """
        catalog = cls.load_catalog_frame()
        purposes: pd.Series = catalog["purpose"]
        return purposes.dropna().unique().tolist()


    @classmethod
    def list_all_datasets(cls) -> list[str]:
        """
        Fast dataset listing without crawling instances.
        Returns sorted keys from Dataset enum.
        """
        return sorted([d.value.name for d in Dataset])


    @classmethod
    def list_datasets(cls, purpose: str) -> list[str]:
        """
        Return sorted list of datasets for a given purpose.
        """
        catalog = cls.load_catalog_frame()
        datasets: pd.Series = catalog.loc[catalog["purpose"] == purpose, "dataset"]
        return datasets.dropna().unique().tolist()


    @classmethod
    def list_instances(cls, dataset: Dataset) -> list[str]:
        """
        Return sorted list of instances for a given dataset.
        """
        catalog = cls.load_catalog_frame()
        dataset_name = dataset.value.name
        instance_series: pd.Series = catalog.loc[catalog["dataset"] == dataset_name, "instance"]
        instances = instance_series.dropna().unique().tolist()
        instances.sort()
        return instances


    @classmethod
    def get_catalog_tree(cls, force_refresh: bool = False) -> dict[str, dict[str, list[str]]]:
        """
        Return a nested mapping {purpose: {dataset: [instances...]}}.
        This is useful for building a collapsible catalog view.
        """
        df: CatalogDF = cls.load_catalog_frame(force_refresh=force_refresh)
        tree: dict[str, dict[str, list[str]]] = {}
        purpose_series: pd.Series = df["purpose"]
        for purpose in purpose_series.dropna().unique():
            view: pd.DataFrame = df.loc[df["purpose"] == purpose]
            dataset_series: pd.Series = view["dataset"]
            datasets: ndarray= dataset_series.dropna().unique()
            tree[purpose] = {}
            for dts in datasets:
                inst_series: Any = view.loc[view["dataset"] == dts, "instance"]
                insts = inst_series.dropna().unique().tolist()
                insts.sort()
                tree[purpose][dts] = insts

        return tree




    # ================ Dashboard bundle ================
    @classmethod
    def load_catalog_frame(cls, force_refresh: bool = False) -> CatalogDF:
        """
        Load and return catalog dataframe:
        columns: ['purpose', 'dataset', 'instance']
        """
        if force_refresh or cls.catalog_cache is None:
            cls.catalog_cache = transform_catalog()
        return cls.catalog_cache.copy()


    @staticmethod
    @lru_cache(maxsize=16)
    def load_dataset_bundle(dataset: Dataset) -> DatasetBundle:
        """
        Load and return normalized attempts and scoreboard dataframes for a dataset.
        """
        dataset_name = dataset.value.name
        attempts_path = path_router(dataset_name, "record")
        scoreboard_path = path_router(dataset_name, "summarize")

        attempts_df = transform_attempts_df(attempts_path)
        scoreboard_df = transform_scoreboard_df(scoreboard_path)
        return (attempts_df, scoreboard_df)


    @classmethod
    def load_dashboard_bundle(cls, dataset: Dataset) -> DashboardBundle:
        """
        Return a minimal bundle of dfs needed by the dashboard for one dataset:
        - attempts: processed attempts record taken from the attempts CSV
        - scoreboard: processed scoreboard record taken from the summarize CSV
        - recent_attempts: transform_attempts_summary(attempts, n=10)
        - solver_summary: transform_solvers_summary(attempts)
        - best_attempts: transform_top_best_attempts(attempts, n=3)
        - solver_leaderboard: transform_solver_leaderboard(scoreboard)
        - solver_recent_hyperparams: transform_solver_config(attempts, limit=3)

        This is useful for building the main dashboard view of the dataset.
        Further instance-scoped charts can be built via instance-scoped APIs.
        """
        try:
            attempts_df, scoreboard_df = cls.load_dataset_bundle(dataset)
        except KeyError as exc:
            raise ValueError(f"Unknown dataset '{dataset.value.name}'") from exc

        attempts_df = attempts_df.copy()
        scoreboard_df = scoreboard_df.copy()

        # Build only with existing helpers (no undefined calls)
        best_attempts_df = transform_top_best_attempts(attempts_df, n=3)
        recent_attempts_df = transform_attempts_summary(attempts_df, n=10)
        solver_summary_df = transform_solvers_summary(attempts_df)
        solver_leaderboard_df = transform_solver_leaderboard(scoreboard_df)
        solver_recent_hparams_df = transform_solver_config(attempts_df, limit=3)

        return {
            "attempts": attempts_df,
            "scoreboard": scoreboard_df,
            "best_attempts": best_attempts_df,
            "recent_attempts": recent_attempts_df,
            "solver_summary": solver_summary_df,
            "solver_leaderboard": solver_leaderboard_df,
            "solver_recent_hyperparams": solver_recent_hparams_df,
        }




    # ================ Instance bundle ================
    @classmethod
    def load_instance_recent(
            cls, dataset: Dataset, instance: str, n: int = 3
        ) -> pd.DataFrame:
        """
        Return most recent successful attempts for a given instance:
        transform_top_recent_attempts(instance, attempts, n)
        """
        attempts_df, _ = cls.load_dataset_bundle(dataset)
        return transform_top_recent_attempts(instance, attempts_df, n=n)


    @classmethod
    def load_recent_vs_best(
            cls, dataset: Dataset, instance: str, n: int = 3
        ) -> pd.DataFrame:
        """
        Return recent vs best attempts for a given instance:
        """

        attempts_df, scoreboard_df = cls.load_dataset_bundle(dataset)
        recent_df = transform_top_recent_attempts(instance, attempts_df, n=n)
        return transform_recent_vs_best(instance, recent_df, scoreboard_df, n=n)


    @classmethod
    def load_instance_bundle(
            cls, dataset: Dataset, instance: str, n: int = 3
        ) -> InstanceBundle:
        """
        Return a minimal bundle of dfs needed by instance-scoped charts:
        - instance_recent: transform_top_recent_attempts(instance, attempts, n)
        - instance_recent_vs_best: transform_recent_vs_best(
            instance, instance_recent, scoreboard, n)
        """
        attempts_df, scoreboard_df = cls.load_dataset_bundle(dataset)
        instance_recent = transform_top_recent_attempts(instance, attempts_df, n=n)
        instance_recent_vs_best = transform_recent_vs_best(
            instance, instance_recent, scoreboard_df, n=n
        )
        return {
            "instance_recent": instance_recent,
            "instance_recent_vs_best": instance_recent_vs_best,
        }




    # ---------- Misc convenience ----------
    @classmethod
    def load_solver_leaderboard(cls, dataset: Dataset) -> pd.DataFrame:
        """Convenience: leaderboard only."""
        _, scoreboard_df = cls.load_dataset_bundle(dataset)
        return transform_solver_leaderboard(scoreboard_df)


    @classmethod
    def load_solver_recent_config(cls, dataset: Dataset, limit: int = 3) -> pd.DataFrame:
        """Convenience: recent solver configs only."""
        attempts_df, _ = cls.load_dataset_bundle(dataset)
        return transform_solver_config(attempts_df, limit=limit)




    # ================ Visualizer loaders ================
    @classmethod
    def load_instance_nodes(
            cls,
            dataset: Dataset | None = None,
            instance: str | None = None,
            content: str | None = None
        ) -> NodesDF:
        """
        Load instance data and transform to nodes DataFrame for visualizer.
        
        Supports two modes:
        - File mode: provide dataset and instance names
        - Content mode: provide raw .sarp content string
        
        Args:
            dataset: Dataset enum (for file mode)
            instance: Instance name (for file mode)
            content: Raw .sarp file content (for content mode)
        
        Returns:
            NodesDF with columns [id, x, y, demand, type_id, type_label]
        """
        if content is not None:
            # Content mode: use transformer convenience function
            return transform_instance_content(content)
        elif dataset is not None and instance is not None:
            # File mode: parse from file, then transform
            file_path = path_router(dataset.value.name, "readfile", filename=instance)
            problem = extract_sarp_content(file_path)
            return transform_nodes(problem)
        else:
            raise ValueError("Must provide either (dataset, instance) or content")


    @classmethod
    def load_solution_edges(
            cls,
            nodes_df: NodesDF,
            default_depot: int = 0,
            content: str | None = None
        ) -> tuple[EdgesDF, int | None]:
        """
        Load solution data and transform to edges DataFrame for visualizer.
        
        Supports two modes:
        - Warehouse mode: load from the main solution warehouse (route.sol)
        - Content mode: provide raw .sol content string
        
        Args:
            nodes_df: Nodes DataFrame for coordinate lookup
            default_depot: Default depot node ID
            content: Raw .sol file content (for content mode), or None to load from warehouse
        
        Returns:
            - EdgesDF: DataFrame with route edge data for visualizer
            - int | None: Reported cost from solution file
        """
        if content is not None:
            # Content mode: use transformer convenience function
            if not content.strip():
                return pd.DataFrame(), None
            return transform_solution_content(content, nodes_df, default_depot)
        else:
            # Warehouse mode: load from route.sol using file parser
            try:
                solution_data = extract_sol_content(SOLUTION_WAREHOUSE_PATH)
            except FileNotFoundError:
                return pd.DataFrame(), None
            return transform_edges(solution_data, nodes_df, default_depot)



    @classmethod
    def load_visualizer_bundle(
            cls,
            dataset: Dataset | None = None,
            instance: str | None = None,
            instance_content: str | None = None,
            solution_content: str | None = None
        ) -> VisualizerBundle:
        """
        Load visualizer's bundle for visualization.

        Supports two modes:
        - File mode: provide dataset and instance names (loads from files)
        - Content mode: provide raw content strings (for paste mode)

        Args:
            dataset: Dataset enum (for file mode)
            instance: Instance name (for file mode)
            instance_content: Raw .sarp file content (for content mode)
            solution_content: Raw .sol file content (for content mode, or None, from warehouse)

        Returns:
            VisualizationBundle with keys: 'nodes_df', 'edges_df', 'reported_cost'
        """
        # Load instance nodes
        if instance_content is not None:
            nodes_df = cls.load_instance_nodes(content=instance_content)
        elif dataset is not None and instance is not None:
            nodes_df = cls.load_instance_nodes(dataset=dataset, instance=instance)
        else:
            raise ValueError("Must provide either (dataset, instance) or instance_content")

        # Depot is always node 0 in canonical ordering
        default_depot = 0

        # Load solution edges
        edges_df, reported_cost = cls.load_solution_edges(
            nodes_df, default_depot, content=solution_content
        )

        return {
            'nodes_df': nodes_df,
            'edges_df': edges_df,
            'reported_cost': reported_cost,
        }




if __name__ == "__main__":
    pass
