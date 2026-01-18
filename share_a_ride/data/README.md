# Data Module

This directory contains the data management, processing, and ETL (Extract, Transform, Load) pipeline for the Share-a-Ride project.

## ETL Architecture

The data pipeline is organized into a formal ETL structure to separate file access, data processing, and application loading:

1.  **Extract (extractor.py)**: Responsible for all physical I/O operations. It reads raw content from .sarp, .sol, and .csv files and handles resource discovery (e.g., crawling datasets to list instances).
2.  **Transform (transformer.py)**: The logic layer that cleans and normalizes raw data. It converts parsed content into structured pandas.DataFrame formats (like NodesDF or EdgesDF) ready for visualization and analysis.
3.  **Load (loader.py)**: The primary API surface for the application layer. It orchestrates the extractor and transformer, implements caching mechanisms (LRU), and provides high-level "Data Bundles" tailored for Dashboard and Visualizer components.

## Module Overview

### Core ETL Layers
- [extractor.py](extractor.py): Physically extracts raw data and listed resources from the filesystem.
- [transformer.py](transformer.py): Transforms raw data structures into normalized DataFrames.
- [loader.py](loader.py): Provides cached data loading and bundling for the application.

### Processing & Analysis
- [executor.py](executor.py): Orchestrates solver execution on instances/datasets and records raw results to *-attempts.csv.
- [summarizer.py](summarizer.py): Processes attempts records to generate *-scoreboard.csv summaries, tracking performance improvements.

### Helpers & Metadata
- [router.py](router.py): Centralized path management mapping actions and datasets to physical file locations.
- [parser.py](parser.py): Grammar-level parsing logic for SARP instance formats and solution files.
- [classes.py](classes.py): Shared domain models, constants, and column definitions.


## Data Structure

The data is organized hierarchically by purpose:
share_a_ride/data/{purpose}/{dataset}/{instance}.sarp

- **Purpose**: High-level categorization (Sanity, Benchmark, Test, Train, Val).
- **Dataset**: A collection of instances (e.g., Li, Solomon, Golden).
- **Instance**: Individual SARP problem files.
- **Attempts/Scoreboard**: Execution records and summaries stored within each dataset folder.

## Support for Application

The data module provides the backbone for the Streamlit-based application ([app.py](../app/app.py)):

- **Dashboard Support**: `DataLoader.load_dashboard_bundle()` provides the `DataBundle` required by [dashboard.py](../app/dashboard.py), including historical performance, solver leaderboards, and hyperparameter trends.
- **Visualizer Support**: `DataLoader.load_visualizer_bundle()` provides the `VisualizationBundle` for [visualizer.py](../app/visualizer.py), containing canonical node coordinates and solution route geometries.
- **Performance**: Heavy use of `functools.lru_cache` in the Load layer ensures that switching between datasets in the UI is nearly instantaneous, as redundant disk I/O and data normalization are avoided.

## Principles and Contribution Guidelines

To maintain the integrity of the data pipeline, please follow these design principles:

1.  **Strict ETL Separation**:
    -   **Extract**: Keep file-system specifics and raw reading in `extractor.py`.
    -   **Transform**: Keep pandas logic, data cleaning, and data-type normalization in `transformer.py`.
    -   **Load**: Keep application-specific bundling and caching in `loader.py`.
2.  **Stateless Extraction**: Extraction functions should be pure and stateless. They should take a path/content and return a dictionary or raw DataFrame.
3.  **Normalized DataFrames**: All transformation functions must return standard `pd.DataFrame` objects using the type aliases and column definitions (e.g., `NodesDF`, `ATTEMPT_DF_COLUMNS`) defined in `transformer.py` and `classes.py`.
4.  **Enum-First API**: Public methods in the Load layer should prefer taking `Dataset` enum members rather than raw strings to ensure type safety across the application.
5.  **Canonical Ordering**: Always maintain the project's canonical node ordering: `Depot (0)` -> `Passenger Pickups` -> `Parcel Pickups` -> `Passenger Dropoffs` -> `Parcel Dropoffs`. This ensures that solvers, metrics, and visualizers remain interoperable.
6.  **Fail Fast in Data, but Safe in Application**: Missing data should be raised as early as possible, while
extraction and transformation functions should handle missing files or empty CSVs gracefully by returning empty DataFrames with correct schemas.
