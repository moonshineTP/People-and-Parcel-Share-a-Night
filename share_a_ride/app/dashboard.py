"""
Streamlit dashboard view for the main app.
To run this dashboard:
    streamlit run share_a_ride/app/dashboard.py
"""

import pandas as pd
import altair as alt
import streamlit as st

# Adjust import based on how the script is run. 
try:
    from share_a_ride.data.loader import DataLoader
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from share_a_ride.data.loader import DataLoader


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

    def __init__(self):
        st.set_page_config(page_title="Share-a-Ride Dashboard", layout="wide")
        st.title("Share-a-Ride Dashboard")
        
        # State
        self.bundle = None
        self.selected_dataset = None
        self.selected_instance = None

    def run(self):
        """Main execution entry point for the dashboard."""
        self._build_catalog()
        self.refresh()
        
        if self.selected_dataset:
            self._on_instance_selected()
            st.divider()
            self._on_solver_selected()
        else:
            st.info("Please select a dataset to view the dashboard.")

    def _build_catalog(self):
        """Build the dataset selection catalog with collapsible controls."""
        st.sidebar.header("Dataset Catalog")

        if st.sidebar.button("Refresh Data"):
            DataLoader.invalidate_caches()
            st.cache_data.clear()
            st.success("Cache cleared. Reloading...")
            st.rerun()

        catalog_tree = DataLoader.get_catalog_tree()

        # Purpose Selection
        purposes = sorted(catalog_tree.keys()) if catalog_tree else []
        selected_purpose = st.sidebar.selectbox("Purpose", purposes)

        # Dataset Selection
        datasets = []
        if selected_purpose and selected_purpose in catalog_tree:
            datasets = sorted(catalog_tree[selected_purpose].keys())
        
        self.selected_dataset = st.sidebar.selectbox("Dataset", datasets)

        # Instance Selection
        instances = []
        if selected_purpose and self.selected_dataset and self.selected_dataset in catalog_tree[selected_purpose]:
            instances = catalog_tree[selected_purpose][self.selected_dataset]
        
        self.selected_instance = st.sidebar.selectbox("Instance", instances)

    def refresh(self):
        """Refresh the dashboard data and visualizations based on the selected dataset."""
        if not self.selected_dataset:
            return

        try:
            self.bundle = DataLoader.load_dataset_bundle(self.selected_dataset)
        except ValueError as e:
            st.error(f"Error loading dataset: {e}")
            self.bundle = None

    def _on_instance_selected(self):
        """Handle instance selection events to update instance-specific views."""
        st.markdown("## Instance Performance")
        
        if not self.selected_instance:
            st.info("Select an instance to see detailed performance charts.")
            return
            
        if not self.bundle:
            return

        try:
            # Load instance specific data
            inst_bundle = DataLoader.load_instance_bundle(self.selected_dataset, self.selected_instance, n=5)
            
            # Filter dataset-level best attempts for the specific instance
            ds_best = self.bundle.get("best_attempts")
            best_df = None
            if isinstance(ds_best, pd.DataFrame) and not ds_best.empty:
                best_df = ds_best.loc[ds_best["instance"] == self.selected_instance].copy()
            
            recent_df = inst_bundle.get("instance_recent_vs_best")

            col1, col2 = st.columns(2)
            
            with col1:
                self._build_instance_best_records(best_df)

            with col2:
                self._build_instance_recent_attempts(recent_df)

        except Exception as e:
            st.error(f"Error loading instance data: {e}")

    def _build_instance_best_records(self, df: pd.DataFrame | None):
        """Build the per-instance best attempts records."""
        st.markdown("#### Best vs Recent Attempts")
        chart = self._render_best_attempts_chart(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data for best attempts.")

    def _build_instance_recent_attempts(self, df: pd.DataFrame | None):
        """Build the per-instance recent attempts plot."""
        st.markdown("#### Recent Attempts (Cost over Time)")
        chart = self._render_recent_attempts_chart(df)
        if chart:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("No data for recent attempts.")

    def _on_solver_selected(self):
        """Handle solver selection events to update solver-specific views."""
        # Note: In this architecture, this updates the general solver views 
        # (Summary, Leaderboard, Configs) which are relevant context for any selected solver.
        st.markdown("## Solver Performance")
        
        if not self.bundle:
            return

        tab1, tab2, tab3 = st.tabs(["Summary", "Leaderboard", "Configurations"])

        with tab1:
            self._build_solver_table()

        with tab2:
            self._build_solver_best_pie_chart()

        with tab3:
            self._build_solver_config_table()

    def _build_solver_table(self):
        """Build the per-solver performance summary table."""
        st.markdown("### Performance Summary")
        solver_summary = self.bundle.get("solver_summary")
        if solver_summary is not None and not solver_summary.empty:
            # Format columns for display
            display_df = solver_summary.copy()
            if "success_rate" in display_df.columns:
                display_df["success_rate"] = display_df["success_rate"].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_df, use_container_width=True)
        else:
            st.write("No solver summary data available.")

    def _build_solver_best_pie_chart(self):
        """Build the per-solver best-known solutions pie chart."""
        st.markdown("### Best-known Solutions Share")
        solver_leaderboard = self.bundle.get("solver_leaderboard")
        col_pie, col_spacer = st.columns([1, 2])
        with col_pie:
            chart = self._render_solver_leaderboard_pie(solver_leaderboard)
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("No leaderboard data available.")

    def _build_solver_config_table(self):
        """Build the per-solver recent configurations table."""
        st.markdown("### Recent Configurations")
        solver_configs = self.bundle.get("solver_recent_hyperparams")
        if solver_configs is None:
             try:
                solver_configs = DataLoader.load_solver_recent_config(self.selected_dataset, limit=3)
             except Exception:
                solver_configs = None
        
        if solver_configs is not None and not solver_configs.empty:
            st.dataframe(solver_configs, use_container_width=True)
        else:
            st.write("No configuration data available.")

    # ============= Static Chart Rendering Helpers ============
    
    @staticmethod
    def _render_best_attempts_chart(df: pd.DataFrame | None):
        """Render the best-attempts bar chart using Altair."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        work = df.copy()
        if "instance" not in work.columns:
            return None

        # Filter invalid costs
        work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
        work = work.dropna(subset=["cost"])
        if work.empty:
            return None

        # Create a rank column for coloring (Top 1, 2, 3)
        work['rank'] = work.groupby('instance')['cost'].rank(method='first').astype(int)
        
        chart = alt.Chart(work).mark_bar().encode(
            x=alt.X('solver:N', axis=alt.Axis(labelAngle=-45, title="Solver")),
            y=alt.Y('cost:Q', title="Cost"),
            color=alt.Color('rank:O', scale=alt.Scale(scheme='category10'), legend=alt.Legend(title="Rank")),
            tooltip=['instance', 'solver', 'cost', 'attempt_id']
        ).properties(
            title="Top Best Attempts per Solver"
        )

        return chart

    @staticmethod
    def _render_recent_attempts_chart(df: pd.DataFrame | None):
        """Render the recent-attempts line chart using Altair."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None

        work = (
            df.copy()
            .dropna(subset=["timestamp", "cost"])
            .sort_values("timestamp", ascending=True)
        )
        work["cost"] = pd.to_numeric(work["cost"], errors="coerce")
        if work.empty:
            return None

        # Determine best cost for the rule line
        best_cost = None
        if "cost_gap" in work.columns:
            zero_gap_rows = work.loc[work["cost_gap"] == 0]
            if not zero_gap_rows.empty:
                best_cost = float(zero_gap_rows["cost"].iloc[0])
        
        if best_cost is None and not work["cost"].empty:
            best_cost = float(work["cost"].min())

        # Base line chart
        line = alt.Chart(work).mark_line(point=True).encode(
            x=alt.X('timestamp:T', title="Time"),
            y=alt.Y('cost:Q', title="Cost", scale=alt.Scale(zero=False)),
            tooltip=['timestamp', 'cost', 'solver', 'status']
        )

        # Threshold rule
        if best_cost is not None:
            rule = alt.Chart(pd.DataFrame({'cost': [best_cost]})).mark_rule(color='green', strokeDash=[5, 5]).encode(
                y='cost:Q',
                tooltip=alt.value(f"Best: {best_cost}")
            )
            return (line + rule).properties(title="Recent Attempts History")
        
        return line.properties(title="Recent Attempts History")

    @staticmethod
    def _render_solver_leaderboard_pie(df: pd.DataFrame | None):
        """Render solver leaderboard pie chart using Altair."""
        if df is None or df.empty:
            return None

        try:
            # Prepare data: need simple summary
            data = df.copy()
            data["num_instances"] = pd.to_numeric(data["num_instances"], errors="coerce")
            data = data[data["num_instances"] > 0]
        except Exception:
            return None

        if data.empty:
            return None

        base = alt.Chart(data).encode(
            theta=alt.Theta("num_instances", stack=True),
            color=alt.Color("solver"),
            tooltip=["solver", "num_instances"]
        )

        pie = base.mark_arc(outerRadius=120)
        text = base.mark_text(radius=140).encode(
            text=alt.Text("num_instances"),
            order=alt.Order("num_instances", sort="descending")
        )

        return (pie + text).properties(title="Best-known Solutions Share")


if __name__ == "__main__":
    dashboard = ShareARideDashboard()
    dashboard.run()
