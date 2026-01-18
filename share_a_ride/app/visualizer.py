"""
Streamlit solution visualizer for the Share-a-Ride app.
To run the app, use:
    streamlit run share_a_ride/app/visualizer.py
"""
import sys
from pathlib import Path

# Add root path to sys.path for module imports
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from typing import Optional

import pandas as pd
import altair as alt
import streamlit as st

from share_a_ride.data.classes import Dataset
from share_a_ride.data.loader import DataLoader
from share_a_ride.solvers.classes import SolverName, SolverMode
from share_a_ride.data.executor import try_instance
from share_a_ride.data.router import path_router
from share_a_ride.data.extractor import extract_sarp_core
from share_a_ride.data.parser import parse_sarp_content, parse_sol_content
from share_a_ride.data.transformer import transform_nodes, transform_edges




class ShareARideSolutionVisualizer:
    """
    Visualizer view for VRP/SARP solutions using Streamlit and Altair.
    
    This is a complete view component (like Dashboard) that handles:
    1. UI controls for data input (file selection or text paste)
    2. Data loading via DataLoader
    3. Visualization rendering
    
    It serves 4 main display functions:
    1. Visualize Nodes (Depot, Pickup, Delivery) on a 2D plane.
    2. Draw Routes differentiating vehicles.
    3. Show directionality using arrow markers.
    4. Display contextual info via tooltips and summary metrics.
    """

    def __init__(self):
        # State
        self.selected_dataset: Optional[str] = None
        self.selected_instance: Optional[str] = None


    def run(self):
        """Main execution entry point for the visualizer view."""
        st.title("Solution Visualizer")

        # Build input controls
        input_mode = st.radio(
            "Input Mode",
            ["Select from Dataset", "Paste Content", "Run Solver"],
            horizontal=True
        )

        if input_mode == "Select from Dataset":
            self._run_dataset_mode()
        elif input_mode == "Paste Content":
            self._run_paste_mode()
        else:
            self._run_solver_mode()


    def _run_dataset_mode(self):
        """Run visualizer with dataset/instance selection."""
        # Dataset and instance selection
        col1, col2 = st.columns(2)

        with col1:
            datasets = DataLoader.list_all_datasets()
            self.selected_dataset = st.selectbox(
                "Select Dataset",
                options=[""] + datasets,
                index=0
            )

        with col2:
            if self.selected_dataset:
                ds_enum = Dataset.from_str(self.selected_dataset)
                instances = DataLoader.list_instances(ds_enum)
                self.selected_instance = st.selectbox(
                    "Select Instance",
                    options=[""] + instances,
                    index=0
                )
            else:
                st.selectbox("Select Instance", options=[""], disabled=True)

        # Load and render
        if self.selected_dataset and self.selected_instance:
            try:
                ds_enum = Dataset.from_str(self.selected_dataset)
                bundle = DataLoader.load_visualizer_bundle(
                    dataset=ds_enum,
                    instance=self.selected_instance
                )
                self.render(
                    nodes_df=bundle['nodes_df'],
                    edges_df=bundle['edges_df'],
                    reported_cost=bundle['reported_cost']
                )
            except Exception as e:  # pylint: disable=broad-except
                st.error(f"Failed to load data: {e}")
        else:
            st.info("Select a dataset and instance to visualize.")


    def _run_solver_mode(self):
        """Run visualizer with solver execution."""
        st.subheader("Run Solver Experiment")

        c1, c2, c3 = st.columns(3)
        with c1:
            datasets = DataLoader.list_all_datasets()
            sel_ds = st.selectbox("Dataset", options=[""] + datasets, key="run_ds")

        with c2:
            if sel_ds:
                ds_enum = Dataset.from_str(sel_ds)
                instances = DataLoader.list_instances(ds_enum)
                sel_inst = st.selectbox("Instance", options=[""] + instances, key="run_inst")
            else:
                sel_inst = st.selectbox("Instance", options=[""], disabled=True, key="run_inst")

        with c3:
            # Solver selection
            solvers = [s.value for s in SolverName]
            sel_solver = st.selectbox("Solver", options=solvers, key="run_solver")

        if st.button("Run Solver", type="primary", disabled=not (sel_ds and sel_inst and sel_solver)):
            with st.spinner("Solving..."):
                try:
                    ds_enum = Dataset.from_str(sel_ds)
                    # 1. Get raw problem
                    inst_path = path_router(sel_ds, "readfile", sel_inst)
                    problem = extract_sarp_core(inst_path)

                    # 2. Write problem to run.inp
                    with open(inst_path, 'r', encoding='utf-8') as f:
                        raw_content = f.read()

                    with open("share_a_ride/data/run.inp", "w", encoding='utf-8') as f:
                        f.write(raw_content)

                    # 3. Run Solver
                    sol, gap = try_instance(
                        dataset=ds_enum,
                        inst_name=sel_inst,
                        problem=problem,
                        solver_name=SolverName(sel_solver),
                        solver_mode=SolverMode.STANDARD,
                        verbose=True
                    )

                    if sol:
                        st.success(f"Solution found! Gap: {gap if gap is not None else 'N/A'}")

                        # 4. Write solution to run.out
                        sol_str = self._serialize_solution(sol)
                        with open("share_a_ride/data/run.out", "w", encoding='utf-8') as f:
                            f.write(sol_str)

                        # 5. Visualize
                        # Read run.inp and run.out
                        with open("share_a_ride/data/run.inp", "r", encoding='utf-8') as f:
                            inp_text = f.read()
                        with open("share_a_ride/data/run.out", "r", encoding='utf-8') as f:
                            out_text = f.read()

                        # Transform
                        instance_content = parse_sarp_content(inp_text)
                        nodes_df = transform_nodes(instance_content)

                        solution_content = parse_sol_content(out_text)
                        edges_df, reported_cost = transform_edges(solution_content, nodes_df)

                        self.render(nodes_df, edges_df, reported_cost)

                    else:
                        st.error("No solution found.")

                except Exception as e:  # pylint: disable=broad-except
                    st.error(f"Error executing solver: {e}")


    def _serialize_solution(self, solution) -> str:
        """
        Helper to convert Solution object to .sol format string.
        
        Note: Following TSPLIB convention, depot is EXCLUDED from the route string.
        The parser/transformer will inject depot at start/end when building edges.
        """
        lines = []
        if hasattr(solution, 'max_cost'):
            lines.append(f"Cost {solution.max_cost}")

        for i, route in enumerate(solution.routes):
            # Exclude depot (0) at start and end - TSPLIB convention
            path = list(route)
            if path and path[0] == 0:
                path = path[1:]
            if path and path[-1] == 0:
                path = path[:-1]
            # Format: Route #1: 1 2 3 (no depot)
            path_str = " ".join(str(int(n)) for n in path)
            lines.append(f"Route #{i+1}: {path_str}")

        return "\n".join(lines)


    def _run_paste_mode(self):
        """Run visualizer with pasted content."""
        st.write("Paste your Instance (SARP format) and Solution content below.")

        col1, col2 = st.columns(2)
        with col1:
            instance_input = st.text_area(
                "Instance Content",
                height=300,
                help="Paste content of .sarp file here"
            )
        with col2:
            solution_input = st.text_area(
                "Solution Content",
                height=300,
                help="Paste solution route text here"
            )

        if st.button("Visualize", type="primary"):
            if instance_input:
                try:
                    # Load visualization data from content via DataLoader
                    bundle = DataLoader.load_visualizer_bundle(
                        instance_content=instance_input,
                        solution_content=solution_input if solution_input else None
                    )

                    # Render visualization
                    self.render(
                        nodes_df=bundle['nodes_df'],
                        edges_df=bundle['edges_df'],
                        reported_cost=bundle['reported_cost']
                    )

                except Exception as e:  # pylint: disable=broad-except
                    st.error(f"Failed to parse input: {e}")
            else:
                st.error("Instance content is required.")




    # ================ Rendering Entry Point ================
    def render(
            self,
            nodes_df: pd.DataFrame,
            edges_df: pd.DataFrame | None = None,
            reported_cost: int | None = None
        ):
        """
        Main entry point to render the visualization.
        
        Args:
            nodes_df: DataFrame with columns [id, x, y, demand, type_id, type_label]
                      The depot is identified by type_label == "Depot" (always node id 0).
            edges_df: DataFrame with route edges (optional), columns include
                      [vehicle_id, order, x, y, mid_x, mid_y, angle, distance]
            reported_cost: Cost value from solution file (optional)
        """
        if nodes_df.empty:
            st.warning("No node data to display.")
            return

        # Ensure edges_df is a DataFrame (not None)
        if edges_df is None:
            edges_df = pd.DataFrame()

        # Draw Chart
        st.markdown("### Solution Visualization")
        chart = self._build_altair_chart(nodes_df, edges_df)
        st.altair_chart(chart, width='stretch')

        # Metrics (if solution exists)
        if not edges_df.empty:
            self._render_metrics(edges_df, reported_cost)


    def _render_metrics(self, edge_df: pd.DataFrame, reported_cost: int | None):
        """Display summary metrics based on the calculated edges."""

        # Calculate total distance based on Euclidean distance of segments
        total_dist = edge_df["distance"].sum()
        num_vehicles = edge_df["vehicle_id"].nunique()

        # Display metrics in 3 columns
        cols = st.columns(3)
        cols[0].metric("Calculated Distance (Euclidean)", f"{total_dist:,.2f}")
        cols[1].metric("Vehicles Used", f"{num_vehicles}")
        if reported_cost is not None:
            cols[2].metric("Reported Cost (from file)", f"{reported_cost:,.2f}")


    def _build_altair_chart(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> alt.LayerChart:
        """
        Compose the Altair chart with Layers: Routes -> Arrows -> Nodes.
        """

        # //// Layer 1: Routes (Lines)
        # We need edges data: x, y, x2, y2, vehicle_id
        routes_layer = alt.Chart(pd.DataFrame()).mark_line()
        arrows_layer = alt.Chart(pd.DataFrame()).mark_point()

        # Visualize
        if not edges_df.empty:
            # Lines
            routes_layer = alt.Chart(edges_df).mark_line(
                opacity=0.8,
                strokeWidth=2
            ).encode(
                x=alt.X('x:Q', axis=None),
                y=alt.Y('y:Q', axis=None),
                order='order:Q',
                color=alt.Color('vehicle_id:N', legend=alt.Legend(title="Vehicle")),
                tooltip=[
                    alt.Tooltip('vehicle_id:N', title='Vehicle'),
                    alt.Tooltip('distance:Q', title='Seg Dist', format='.1f')
                ]
            )


        # //// Layer 2: Nodes (Points)
        # Base node layer
        nodes_base = alt.Chart(nodes_df).encode(
            x=alt.X('x:Q', title='X Coord'),
            y=alt.Y('y:Q', title='Y Coord'),
            tooltip=[
                alt.Tooltip('id:N', title='Node ID'),
                alt.Tooltip('type_label:N', title='Type'),
                alt.Tooltip('demand:Q', title='Demand')
            ]
        )

        # Draw Depots (Squares, Magenta) - single depot per instance (always node 0)
        depots = nodes_base.transform_filter(
            alt.datum.type_label == 'Depot'
        ).mark_square(
            size=200, color='darkmagenta', opacity=1
        )

        # Passenger Pickup: Left half-circle (wedge pointing left) - Blue
        # SVG path for left half-circle: arc from top to bottom on left side
        left_half_circle = "M 0 -1 A 1 1 0 0 0 0 1 L 0 -1 Z"
        pickups_p = nodes_base.transform_filter(
            alt.datum.type_label == 'pickP'
        ).mark_point(
            shape=left_half_circle, size=200, filled=True, opacity=0.9
        ).encode(
            color=alt.value("#1f77b4")  # Blue
        )

        # Passenger Dropoff: Right half-circle (wedge pointing right) - Blue (same color, paired)
        right_half_circle = "M 0 -1 A 1 1 0 0 1 0 1 L 0 -1 Z"
        deliveries_p = nodes_base.transform_filter(
            alt.datum.type_label == 'dropP'
        ).mark_point(
            shape=right_half_circle, size=200, filled=True, opacity=0.9
        ).encode(
            color=alt.value("#1f77b4")  # Blue (same as pickup for pairing)
        )

        # Parcel Pickup: Up triangle - Orange
        pickups_l = nodes_base.transform_filter(
            alt.datum.type_label == 'pickL'
        ).mark_point(
            shape="triangle-up", size=160, filled=True, opacity=0.8
        ).encode(
            color=alt.value("#ff7f0e")  # Orange
        )

        # Parcel Dropoff: Down triangle - Orange (same color, paired)
        deliveries_l = nodes_base.transform_filter(
            alt.datum.type_label == 'dropL'
        ).mark_point(
            shape="triangle-down", size=160, filled=True, opacity=0.8
        ).encode(
            color=alt.value("#ff7f0e")  # Orange (same as pickup for pairing)
        )

        # Labels (Node IDs) - Optional, can be cluttered if many nodes
        # Only show for small instances or on hover? Let's show small text always.
        text = nodes_base.mark_text(
            align='left', baseline='middle', dx=7, fontSize=10, color='white'
        ).encode(text='id:N')

        # Combine
        return (
            routes_layer + arrows_layer + text
            + depots + pickups_p + pickups_l + deliveries_p + deliveries_l
        ).interactive()




if __name__ == "__main__":
    visualizer = ShareARideSolutionVisualizer()
    visualizer.run()
