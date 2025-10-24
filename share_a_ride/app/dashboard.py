"""
Tkinter dashboard view container for the main app.
"""

from __future__ import annotations

import math

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

from share_a_ride.data.loader import DataLoader


# NOTE: On Windows, set DPI awareness to avoid blurry UI on high-DPI displays.
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass


class ShareARideDashboard:
    """
    Dashboard view wrapping the data retrived from the Share-a-Ride data loader.

    The dashboard includes the following components and functionalities of 3 domains:
    - Dataset selection controls with collapsible panel and a refresh button.
    - Performance tracking per-instance and per-solver, includes:
        + Per-instance:
            + Bar plot of the best attempts.
            + Line plot of recent attempts and threshold line to the best-known solutions.
        + Per-solver:
            + Solver success rates and the most recent attempts, shown in a table.
            + Pie chart showing number of the best known solutions each solver holds.
    - Manage solver configurations over time: seed, hyperparameters,... shown in a table.

    The dashboard serves these domains according to these main methods:
    - _build_catalog: Build the dataset selection catalog with collapsible controls.
    - _build_instance_best_records: Build the per-instance best attempts records.
    - _build_instance_recent_attempts: Build the per-instance recent attempts plot.
    - _build_solver_table: Build the per-solver performance summary table.
    - _build_solver_best_pie_chart: Build the per-solver best-known solutions pie chart.
    - _build_solver_config_table: Build the per-solver recent configurations table.
    - refresh: Refresh the dashboard data and visualizations based on the selected dataset.
    - _on_instance_selected: Handle instance selection events to update instance-specific views.
    - _on_solver_selected: Handle solver selection events to update solver-specific views.
    """


    # ============= Initialization ============
    def __init__(self, parent: tk.Widget, default_dataset: str = "H") -> None:
        self.parent = parent

        # Container frame exposed to the app
        self.frame = ttk.Frame(parent)
        self.dataset_var = tk.StringVar(value=default_dataset)


        # Scrollable body: Canvas + vertical Scrollbar + inner content frame
        self._canvas = tk.Canvas(
            self.frame, borderwidth=0, highlightthickness=0
        )
        self._vscroll = ttk.Scrollbar(
            self.frame, orient=tk.VERTICAL, command=self._canvas.yview
        )

        self._canvas.configure(yscrollcommand=self._vscroll.set)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._content = ttk.Frame(self._canvas)
        self._content_id = self._canvas.create_window(
            (0, 0), window=self._content, anchor="nw"
        )

        # Keep a friendly alias for builder methods
        self._body = self._content

        # Update scrollregion when content size changes
        def _on_frame_configure(event: tk.Event) -> None:
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))

        # Ensure inner frame width follows canvas width to avoid horizontal scroll
        def _on_canvas_configure(event: tk.Event) -> None:
            try:
                self._canvas.itemconfigure(self._content_id, width=event.width)
            except Exception:
                pass

        # Bind events
        self._content.bind("<Configure>", _on_frame_configure)
        self._canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling (Windows: event.delta multiples of 120)
        def _on_mousewheel(event: tk.Event) -> None:
            try:
                delta_units = int(-1 * (event.delta / 120))
            except Exception:
                delta_units = -1
            self._canvas.yview_scroll(delta_units, "units")

        # Bind on both canvas and content to work when hovering
        self._canvas.bind("<MouseWheel>", _on_mousewheel)
        self._content.bind("<MouseWheel>", _on_mousewheel)

        # Set styles
        ttk.Style().configure("Solver.Treeview", rowheight=40, font=("Segoe UI", 12))
        ttk.Style().configure("Solver.Treeview.Heading", font=("Segoe UI", 12, "bold"))

        # Core navigation/cache attributes should be declared up front
        # to satisfy linters/type checkers and make the object shape explicit.
        self._catalog_df: pd.DataFrame | None = None
        self._catalog_tree: dict[str, dict[str, list[str]]] = {}
        self._purpose_var = tk.StringVar()
        self._instance_var = tk.StringVar()
        self._dataset_bundle: dict[str, pd.DataFrame] | None = None

        # Build dashboard sections
        self._build_catalog_views()
        self._build_instance_views()
        self._build_solver_views()
        self._build_solver_config_views()

        # Defer data loading until after the window is displayed
        self.frame.after(0, self.refresh)





    # ============= Domain 1: Dataset catalog controls ============
    def _build_catalog_views(self) -> None:
        """
        Build collapsible dataset controls backed by the DataLoader catalog APIs.
        
        The catalog panel includes cascading comboboxes for purpose, dataset, 
        and instance selection, along with a refresh button to reload the catalog 
        from the data source.
        
        The workflow is as purpose->dataset->instance, where each selection 
        filters the next level options.
        """

        # -- Container frames for header and collapsible content --
        controls = ttk.Frame(self._body)
        controls.pack(fill=tk.X, padx=20, pady=20)
        content = ttk.Frame(controls)
        self._dataset_controls_expanded = tk.BooleanVar(value=True)


        # ======== Helpers for collapsing and expanding the catalog panel ========
        def set_visibility(visible: bool) -> None:
            self._dataset_controls_expanded.set(visible)
            if visible:
                content.pack(fill=tk.X, pady=(4, 0))
                toggle_btn.configure(text="Dataset ▼")
            else:
                content.pack_forget()
                toggle_btn.configure(text="Dataset ▲")

        def toggle() -> None:
            set_visibility(not self._dataset_controls_expanded.get())


        # -- Header controls: toggle button and refresh trigger --
        toggle_btn = ttk.Button(    # Button to toggle catalog visibility
            controls, text="Dataset ▼", command=toggle
        )
        toggle_btn.pack(side=tk.LEFT)

        refresh_btn = ttk.Button(
            controls, text="Refresh", command=lambda: (
                self._render_catalog_views(True), self.refresh()
                if self.dataset_var.get() else None
            )
        )
        refresh_btn.pack(side=tk.RIGHT)


        # -- Layout of catalog selectors --
        content.columnconfigure(1, weight=1)
        ttk.Label(content, text="Purpose").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(content, text="Dataset").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Label(content, text="Instance").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4
        )

        self._purpose_combo = ttk.Combobox(
            content, textvariable=self._purpose_var, state="readonly"
        )
        self._purpose_combo.grid(row=0, column=1, sticky="ew", pady=4)

        self._dataset_combo = ttk.Combobox(
            content, textvariable=self.dataset_var, state="readonly"
        )
        self._dataset_combo.grid(row=1, column=1, sticky="ew", pady=4)

        self._instance_combo = ttk.Combobox(
            content, textvariable=self._instance_var, state="readonly"
        )
        self._instance_combo.grid(row=2, column=1, sticky="ew", pady=4)


        # -- Bind events to instance methods (render-time updates) --
        self._purpose_combo.bind("<<ComboboxSelected>>", self._on_purpose_change)
        self._dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_change)
        self._instance_combo.bind("<<ComboboxSelected>>", self._on_instance_change)

        # -- Bootstrap catalog state and UI visibility --
        set_visibility(True)
        # Defer catalog rendering to keep UI responsive at startup
        self.frame.after(0, lambda: self._render_catalog_views(force=False))



    def _render_catalog_views(self, force: bool = False) -> None:
        """Render/populate the catalog controls (purposes, datasets, instances).

        Two-stage flow inside render:
        - Load/refresh catalog data (DataLoader caches) when force=True.
        - Use helper updaters to cascade purpose -> dataset -> instance while
          preserving selections when possible.
        """
        # ---- Load/refresh catalog data ----
        try:
            self._catalog_df = DataLoader.load_catalog_frame(force_refresh=force)
            self._catalog_tree = DataLoader.get_catalog_tree(force_refresh=force)
        except Exception as exc:
            messagebox.showerror("Catalog Error", str(exc), parent=self.frame)
            return

        # ---- Drive the cascade using instance helpers ----
        self._update_purposes(self._find_purpose_for_dataset(self.dataset_var.get()))


    # ======== Event handlers for catalog changes (instance level) ========
    def _on_purpose_change(self, *_: object) -> None:
        """Update datasets and instances when purpose changes (no IO)."""
        self._update_datasets(self._purpose_var.get(), keep_current=False)


    def _on_dataset_change(self, *_: object) -> None:
        """Update instances when dataset changes, then refresh dashboard."""
        dataset = self.dataset_var.get()
        self._update_instances(self._purpose_var.get(), dataset, keep_current=False)
        if dataset:
            self.refresh()

    # ======== Catalog utility helpers (instance-level) ========
    def _find_purpose_for_dataset(self, dataset: str | None) -> str | None:
        """Return the purpose for a dataset using the catalog frame, if available."""
        if not dataset or self._catalog_df is None:
            return None
        matches = self._catalog_df.loc[
            self._catalog_df["dataset"] == dataset, "purpose"
        ]
        return matches.iloc[0] if not matches.empty else None

    def _set_combo(self,
            var: tk.StringVar,
            combo: ttk.Combobox,
            values: list[str],
            keep_current: bool
        ) -> str:
        """
        Utility to set combobox values and select a value.
        - Var: The StringVar linked to the combobox.
        - Combo: The ttk.Combobox to update.
        - Values: The list of values to set in the combobox.
        - keep_current: Whether to keep the current selection if still valid.

        Returns the selected value.
        """

        current = var.get()
        next_value = current if (keep_current and current in values) \
            else (values[0] if values else "")
        var.set(next_value)
        combo["values"] = values
        return next_value

    def _update_instances(self,
            purpose: str | None,
            dataset: str | None,
            keep_current: bool = True
        ) -> str:
        """Update instance combo based on purpose/dataset and return selection."""
        instances: list[str] = []
        if purpose and dataset and self._catalog_tree:
            instances = self._catalog_tree.get(purpose, {}).get(dataset, [])
        return self._set_combo(
            self._instance_var, self._instance_combo, instances, keep_current
        )

    def _update_datasets(self,
            purpose: str | None,
            keep_current: bool = True
        ) -> str:
        """Update dataset combo based on purpose, cascade to instances, return dataset selection."""
        # Retrieve datasets for purpose
        datasets: list[str] = []
        if purpose and self._catalog_tree:
            datasets = sorted(self._catalog_tree.get(purpose, {}).keys())

        # Update dataset combo
        ds = self._set_combo(
            self.dataset_var, self._dataset_combo, datasets, keep_current
        )
        self._update_instances(purpose, ds, keep_current=False)
        return ds

    def _update_purposes(self,
            preferred: str | None = None
        ) -> str:
        """Update purpose combo, cascade to datasets, return purpose selection."""
        purposes: list[str] = sorted(self._catalog_tree.keys()) \
            if self._catalog_tree else []

        # Try to preserve the preferred; else preserve current if valid; else pick first
        current = self._purpose_var.get()
        target = (
            preferred if preferred in purposes else (
                current if current in purposes else (
                    purposes[0] if purposes else ""
                )
            )
        )
        pr = self._set_combo(
            self._purpose_var, self._purpose_combo, purposes, keep_current=False
        )

        # Ensure selected matches target if different
        if target and target in purposes and target != pr:
            self._purpose_var.set(target)

        # Cascade
        self._update_datasets(self._purpose_var.get(), keep_current=True)
        return self._purpose_var.get()





    # ============= Domain 2: Per-instance performance views ============
    def _build_instance_views(self) -> None:
        """
        Build the per-instance charts.
        - Bar chart of best vs recent attempts for the selected instance.
        - Line chart of recent attempts with a horizontal line at the best-known cost.
        """
        inst_frame = ttk.LabelFrame(self._body, text="Instance")
        inst_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Best vs recent (bar chart)
        best_container = ttk.Frame(inst_frame)
        best_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        # Fixed plot pixel size to keep aspect ratio consistent
        plot_w, plot_h = 1080, 480
        self._inst_best_fig, self._inst_best_ax = plt.subplots(figsize=(6, 6))
        # Ensure the Matplotlib figure maps to the desired pixel dimensions
        try:
            self._inst_best_fig.set_size_inches(plot_w / self._inst_best_fig.dpi,
                                                 plot_h / self._inst_best_fig.dpi)
        except Exception:
            pass
        self._inst_best_ax.set_title("Best vs recent attempts")
        self._inst_best_ax.set_ylabel("Cost")
        self._inst_best_canvas = FigureCanvasTkAgg(
            self._inst_best_fig, master=best_container
        )
        # Fix the canvas widget size (pixels) and avoid expanding to parent width
        try:
            w = self._inst_best_canvas.get_tk_widget()
            w.configure(width=plot_w, height=plot_h)
            w.pack(padx=8, pady=8)
        except Exception:
            # Fallback to default packing
            self._inst_best_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Recent attempts over time (line chart)
        recent_container = ttk.Frame(inst_frame)
        recent_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        # Match the recent plot size to the best plot for consistent layout
        try:
            self._inst_recent_fig, self._inst_recent_ax = plt.subplots(figsize=(6, 6))
            self._inst_recent_fig.set_size_inches(plot_w / self._inst_recent_fig.dpi,
                                                  plot_h / self._inst_recent_fig.dpi)
        except Exception:
            self._inst_recent_fig, self._inst_recent_ax = plt.subplots(figsize=(6, 6))
        self._inst_recent_ax.set_title("Recent attempts (cost over time)")
        self._inst_recent_ax.set_ylabel("Cost")
        self._inst_recent_canvas = FigureCanvasTkAgg(
            self._inst_recent_fig, master=recent_container
        )
        try:
            w2 = self._inst_recent_canvas.get_tk_widget()
            w2.configure(width=plot_w, height=plot_h)
            w2.pack(padx=8, pady=8)
        except Exception:
            self._inst_recent_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



    def _render_best_attempts_chart(self, df) -> None:
        """
        Render the best-attempts bar chart built from _build_top_best_attempts.
        """

        if not hasattr(self, "_inst_best_ax"):
            return

        # Prepare the chart canvas
        ax = self._inst_best_ax
        ax.clear()
        ax.set_title("Top best attempts per instance")
        ax.set_ylabel("Cost")

        # Helper to draw 'No data' dummy chart
        def draw_no_data(msg: str = "No data") -> None:
            ax.text(0.5, 0.5, msg, ha="center", va="center")
            self._inst_best_canvas.draw_idle()

        if not isinstance(df, pd.DataFrame) or df.empty:
            draw_no_data()
            return

        work = df.copy()
        if work.empty:
            draw_no_data()
            return

        # Prepare chart attributes (expect dataset-level bests with 'instance')
        rank_colors = ["tab:green", "tab:blue", "tab:orange"]
        positions: list[float] = []
        heights: list = []
        colors: list[str] = []
        tick_labels: list[str] = []
        x = 0.0
        gap = 0.8

        # Expect a dataset-level bests DataFrame with an 'instance' column.
        # If that's not present, treat as no data (clarifies contract).
        if "instance" not in work.columns:
            draw_no_data("No dataset-level bests supplied")
            return

        instances = work["instance"].unique()

        # Extract top 3 best attempts per instance and its attributes
        for inst in instances:
            inst_rows = work.loc[work["instance"] == inst].head(3)
            for rank, row in enumerate(inst_rows.itertuples(index=False)):
                positions.append(x)
                heights.append(getattr(row, "cost", None))
                colors.append(rank_colors[min(rank, len(rank_colors) - 1)])
                # row may have solver attribute
                solver_name = getattr(row, "solver", "")
                tick_labels.append(f"{solver_name}\n{inst}")
                x += 1.0

            # add a small gap between instance groups
            x += gap


        # Filter and coerce heights to numeric, skipping invalid entries
        filtered: list[tuple[float, float, str, str]] = []
        for p, h, c, t in zip(positions, heights, colors, tick_labels):
            try:
                num = float(h)
                if not math.isnan(num):
                    filtered.append((p, num, c, t))
            except Exception:
                # Skip entries that cannot be converted to float
                continue

        if not filtered:
            draw_no_data()
            return

        # Unzip filtered data
        positions = [p for (p, _, _, _) in filtered]
        heights = [h for (_, h, _, _) in filtered]
        colors = [c for (_, _, c, _) in filtered]
        tick_labels = [t for (_, _, _, t) in filtered]

        # Plot bars
        ax.bar(positions, heights, color=colors, width=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=10, ha="right", fontsize=10)
        ax.set_xlabel("Solver (grouped by instance)")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.margins(x=0.01)
        for pos, cost in zip(positions, heights):
            ax.text(pos, cost, str(cost), ha="center", va="bottom", fontsize=10)

        # Finalize the chart
        self._inst_best_canvas.draw_idle()



    def _render_recent_attempts_chart(self, df) -> None:
        """
        Render the recent-attempts line chart built from _build_recent_vs_best_attempts.
        """
        if not hasattr(self, "_inst_recent_ax"):
            return

        # Prepare the chart canvas
        ax = self._inst_recent_ax
        ax.clear()
        ax.set_title("Recent attempts (cost over time)")
        ax.set_ylabel("Cost")

        # Helper to draw 'No data' dummy chart
        def draw_no_data(msg: str = "No data") -> None:
            ax.text(0.5, 0.5, msg, ha="center", va="center")
            self._inst_recent_canvas.draw_idle()

        # Handle empty/missing data
        if not isinstance(df, pd.DataFrame) or df.empty:
            draw_no_data()
            return

        # Prepare data
        work = (
            df.copy()
            .dropna(subset=["timestamp", "cost"])
            .sort_values("timestamp", ascending=True, kind="mergesort")
        )
        if work.empty:
            draw_no_data()
            return

        # Plot recent attempts as points/line
        timestamps = work["timestamp"]
        costs = work["cost"]
        ax.plot(timestamps, costs, marker="o", color="tab:blue", label="recent")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.tick_params(axis="x", labelrotation=10)
        ax.set_xlabel("Timestamp")


        # Plot best-known cost as horizontal line
        best_cost = None
        if "cost_gap" in work.columns:
            zero_gap_rows: pd.DataFrame = work.loc[work["cost_gap"] == 0]
            if not zero_gap_rows.empty:
                best_cost = float(zero_gap_rows["cost"].iloc[0])
        if best_cost is None and not costs.empty:
            best_cost = float(costs.min())
        handles, labels = ax.get_legend_handles_labels()
        if best_cost is not None:
            ax.axhline(
                best_cost,
                color="tab:green",
                linestyle="--",
                linewidth=1,
                label="best",
            )
            handles, labels = ax.get_legend_handles_labels()

        # Finalize the chart
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=10)

        self._inst_recent_canvas.draw_idle()



    # -------- Instance events --------
    def _on_instance_change(self, *_: object) -> None:
        """Handle instance combobox change by rendering instance charts."""

        # Handle no data case
        dataset = self.dataset_var.get()
        instance = self._instance_var.get()
        if not dataset or not instance:
            # Clear charts if missing context
            self._render_best_attempts_chart(None)
            self._render_recent_attempts_chart(None)
            return

        # Reload the instance bundle
        try:
            inst_bundle = DataLoader.load_instance_bundle(dataset, instance, n=5)
        except Exception as exc:
            messagebox.showerror("Instance Error", str(exc), parent=self.frame)
            return

        # Determine best-attempts DataFrame for this instance using the
        # dataset-level 'best_attempts' produced by DataLoader.load_dataset_bundle.
        # This DataFrame contains the top best attempts per instance (columns include 'instance').
        best_df = None
        try:
            if isinstance(self._dataset_bundle, dict):
                ds_best = self._dataset_bundle.get("best_attempts")
                if isinstance(ds_best, pd.DataFrame) and not ds_best.empty:
                    best_df = ds_best.loc[ds_best["instance"] == instance].copy()
        except Exception:
            best_df = None

        # Render charts: use the filtered best_df for the bar chart and
        # keep the recent attempts chart using the instance-scoped bundle.
        self._render_best_attempts_chart(best_df)
        self._render_recent_attempts_chart(inst_bundle.get("instance_recent_vs_best"))



    # ============= Domain 2: Per-solver performance views ============
    def _build_solver_views(self) -> None:
        """
        Build the per-solver performance chart and table views.
        Summary table and pie chart.
        """

        # Frame for solver views section
        solver_frame = ttk.LabelFrame(self._body, text="Solvers")
        solver_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Container for table and scrollbars
        table_container = ttk.Frame(solver_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        table_container.config(width=300)
        table_container.pack_propagate(False)

        # Define columns before Treeview
        columns: tuple[str, ...] = (
            "solver",
            "num_attempts",
            "successes",
            "success_rate",
            "unique_instances",
        )

        # Treeview setup
        # Note: `height` controls visible rows not pixel height. Keep it or remove as desired.
        self.solver_tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            height=15,
            style="Solver.Treeview"
        )
        self.solver_tree.grid(row=0, column=0, sticky="nsew")

        # Layout expansion for grid
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        # Define headings and column properties
        for col in columns:
            # Set heading text
            width = 500 if col == "solver" else 300
            self.solver_tree.heading(
                col,
                text=col.replace("_", " ").title()
            )
            self.solver_tree.column(
                col,
                anchor=tk.CENTER,
                width=width,
                minwidth=100,
                stretch=False
            )

        # Scrollbars
        yscroll = ttk.Scrollbar(
            table_container,
            orient=tk.VERTICAL,
            command=self.solver_tree.yview
        )
        xscroll = ttk.Scrollbar(
            table_container,
            orient=tk.HORIZONTAL,
            command=self.solver_tree.xview
        )
        self.solver_tree.configure(
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set
        )
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        # ================== Pie chart ==================
        chart_frame = ttk.Frame(solver_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        solver_w, solver_h = 600, 480
        self._solver_pie_fig, self._solver_pie_ax = plt.subplots(figsize=(5, 5))
        self._solver_pie_ax.axis("equal")
        self._solver_pie_ax.set_title("Best-known solutions by solver")
        self._solver_pie_canvas = FigureCanvasTkAgg(
            self._solver_pie_fig,
            master=chart_frame
        )
        try:
            self._solver_pie_fig.set_size_inches(
                solver_w / self._solver_pie_fig.dpi,
                solver_h / self._solver_pie_fig.dpi
            )
        except Exception:
            pass
        try:
            widget = self._solver_pie_canvas.get_tk_widget()
            widget.configure(width=solver_w, height=solver_h)
            widget.pack(padx=8, pady=8)
        except Exception:
            self._solver_pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)



    def _render_solver_summary_table(self, df) -> None:
        """
        Render/update the solver performance summary table from a dataframe.
        """
        if not hasattr(self, "solver_tree"):
            return

        tree = self.solver_tree

        # Clear existing rows
        for item in tree.get_children():
            tree.delete(item)

        # Nothing else to do if df is None/empty
        if df is None or getattr(df, "empty", True):
            return

        # Insert rows directly from df; minimal formatting for success_rate
        for _, row in df.iterrows():
            rate = row.get("success_rate")
            formatted_rate = ""
            if isinstance(rate, (int, float)) and not math.isnan(rate):
                formatted_rate = f"{rate:.0%}"
            elif rate not in (None, ""):
                formatted_rate = str(rate)

            values = (
                row.get("solver", ""),
                row.get("num_attempts", ""),
                row.get("successes", ""),
                formatted_rate,
                row.get("unique_instances", ""),
            )
            tree.insert("", tk.END, values=values)



    def _render_solver_leaderboard_pie(self, df) -> None:
        """Render/update the solver leaderboard pie chart from a dataframe.

        Expects columns: ['solver', 'num_instances'] from DataLoader.solver_leaderboard.
        Gracefully handles empty or missing data by clearing the chart.
        """
        # Ensure chart objects exist
        if not hasattr(self, "_solver_pie_ax"):
            return
        ax = self._solver_pie_ax
        ax.clear()
        ax.axis("equal")
        ax.set_title("Best-known solutions by solver")

        # Handle empty/missing data
        if df is None or getattr(df, "empty", True):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self._solver_pie_canvas.draw_idle()
            return

        # Extract labels and sizes
        try:
            labels = df.get("solver", [])
            sizes = df.get("num_instances", [])
        except Exception:
            labels, sizes = [], []

        # Convert to lists for matplotlib and filter zeroes
        items = [
            (str(lbl), int(val))
            for lbl, val in zip(list(labels), list(sizes))
            if isinstance(val, (int, float)) and val > 0
        ]
        if not items:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            self._solver_pie_canvas.draw_idle()
            return

        labels, sizes = zip(*items)
        pie_res = ax.pie(
            sizes,
            labels=labels,
            autopct=(lambda p: f"{p:.0f}%" if p >= 5 else ""),
            startangle=90,
            pctdistance=0.8,
        )
        if len(pie_res) == 3:       # Unpack based on return length
            _, texts, autotexts = pie_res
        else:                       # type: ignore[misc]
            _, texts = pie_res
            autotexts = []
        for text in texts:
            text.set_fontsize(10)
        for autot in autotexts:
            autot.set_fontsize(10)

        # Draw the updated chart
        self._solver_pie_canvas.draw_idle()





    # ============ Domain 3: Solver configuration views ============
    def _build_solver_config_views(self) -> None:
        """
        Build the per-solver configuration table view.
        Table contains solver configurations with hyperparameters and seeds.
        """

        # LabelFrame container
        cfg_frame = ttk.LabelFrame(self._body, text="Solver configs")
        cfg_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        # Container for Treeview + scrollbars
        table_container = ttk.Frame(cfg_frame)
        # Fill and expand so the table grows with the window
        table_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        table_container.config(width=800)
        table_container.pack_propagate(False)

        # Define columns to display
        columns: tuple[str, ...] = (
            "solver",
            "attempt_id",
            "instance",
            "seed",
            "time_limit",
            "hyperparams",
        )

        # Treeview widget
        self.solver_config_tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            height=18,  # Visible rows only. More rows will trigger scrolling.
            style="Solver.Treeview",
        )
        self.solver_config_tree.grid(row=0, column=0, sticky="nsew")

        # Expand Treeview to fill cell in grid
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        # Column headers
        for col in columns:
            width = 1800 if col == "hyperparams" \
                else 450 if col == "solver" \
                else 300 if col == "instance" \
                else 180
            self.solver_config_tree.heading(
                col,
                text=col.replace("_", " ").title()
            )
            self.solver_config_tree.column(
                col,
                anchor=tk.CENTER,
                width=width,
                minwidth=100,
                stretch=False
            )


        # Scrollbars: link to Treeview scroll commands
        yscroll = ttk.Scrollbar(
            table_container,
            orient=tk.VERTICAL,
            command=self.solver_config_tree.yview
        )
        xscroll = ttk.Scrollbar(
            table_container,
            orient=tk.HORIZONTAL,
            command=self.solver_config_tree.xview
        )
        self.solver_config_tree.configure(
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set
        )

        # Position scrollbars so they align correctly
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")



    def _render_solver_config_table(self, df) -> None:
        """
        Render/update the solver configuration table from a dataframe.
        """
        if not hasattr(self, "solver_config_tree"):
            return

        tree = self.solver_config_tree

        # Clear existing rows
        for item in tree.get_children():
            tree.delete(item)

        # Handle missing/empty df minimally (no extra processing)
        if df is None or getattr(df, "empty", True):
            return

        # Define columns to display
        columns: tuple[str, ...] = (
            "solver",
            "attempt_id",
            "instance",
            "seed",
            "time_limit",
            "hyperparams",
        )

        # Reconfigure tree columns/headings to match DataFrame
        tree["columns"] = columns
        for col in columns:
            # Widths to show more text without immediate truncation
            width = 1800 if col == "hyperparams" \
                else 450 if col == "solver" \
                else 300 if col == "instance" \
                else 180
            tree.heading(
                col, 
                text=col.replace("_", " ").title()
            )
            tree.column(col, anchor=tk.CENTER, width=width, minwidth=100, stretch=False)

        # Insert rows directly from df without extra formatting
        for _, row in df.iterrows():
            values = [str(row.get(col, "")) for col in columns]
            tree.insert("", tk.END, values=tuple(values))



    # ============= Dashboard refresh ============
    def refresh(self) -> None:
        """
        Refresh dashboard artefacts by loading the selected dataset bundle and
        delegating rendering to the existing view helpers.
        """
        dataset = self.dataset_var.get()

        # Handle no dataset selected
        if not dataset:
            self._render_solver_summary_table(None)
            self._render_best_attempts_chart(None)
            self._render_recent_attempts_chart(None)
            self._render_solver_leaderboard_pie(None)
            self._render_solver_config_table(None)
            return

        # Load dataset bundle
        try:
            bundle = DataLoader.load_dataset_bundle(dataset)
        except ValueError as exc:
            messagebox.showerror("Dataset Error", str(exc), parent=self.frame)
            return
        self._dataset_bundle = bundle


        # Re-render per-solver dashboard (table + pie + configs)
        self._render_solver_summary_table(bundle.get("solver_summary"))
        self._render_solver_leaderboard_pie(bundle.get("solver_leaderboard"))
        cfg_df = bundle.get("solver_recent_hyperparams")
        if cfg_df is None:
            try:
                cfg_df = DataLoader.load_solver_recent_config(dataset, limit=3)
            except Exception:
                cfg_df = None
        self._render_solver_config_table(cfg_df)


        # Re-render instance views for the selected instance
        self._on_instance_change()
