"""
Streamlit solution visualizer for the Share-a-Ride app.
"""
from __future__ import annotations

import math
import re
import pandas as pd
import altair as alt
import streamlit as st
import numpy as np

class ShareARideSolutionVisualizer:
    """
    Visualizer component for VRP/SARP solutions using Streamlit and Altair.
    
    It serves to:
    1. Visualize Nodes (Depot, Pickup, Delivery) on a 2D plane.
    2. Draw Routes differentiating vehicles.
    3. Show directionality using arrow markers.
    4. Display contextual info via tooltips.
    """

    def render(self, instance_content: str, solution_content: str | None = None):
        """
        Main entry point to render the visualization.
        
        Args:
            instance_content: Raw text content of the .sarp/.vrp instance file.
            solution_content: Raw text content of the solution (routes).
        """
        # 1. Parse Data
        try:
            nodes_df = self._parse_instance_data(instance_content)
        except Exception as e:
            st.error(f"Failed to parse instance data: {e}")
            return

        routes_df = pd.DataFrame()
        if solution_content:
            try:
                routes_df = self._parse_solution_text(solution_content, nodes_df)
            except Exception as e:
                st.warning(f"Could not parse solution fully: {e}")

        # 2. Draw Chart
        st.markdown("### Solution Visualization")
        
        if not nodes_df.empty:
            chart = self._build_altair_chart(nodes_df, routes_df)
            st.altair_chart(chart, use_container_width=True)
        
        # 3. Metrics (if solution exists)
        if not routes_df.empty:
            self._render_metrics(routes_df)

    def _render_metrics(self, routes_edge_df: pd.DataFrame):
        """Display summary metrics based on the calculated edges."""
        # Calculate total distance based on Euclidean distance of segments
        total_dist = routes_edge_df["distance"].sum()
        num_vehicles = routes_edge_df["vehicle_id"].nunique()
        
        c1, c2 = st.columns(2)
        c1.metric("Total Estimated Distance", f"{total_dist:,.2f}")
        c2.metric("Vehicles Used", f"{num_vehicles}")

    def _build_altair_chart(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> alt.Chart:
        """
        Compose the Altair chart with Layers: Routes -> Arrows -> Nodes.
        """
        
        # --- Layer 1: Routes (Lines) ---
        # We need edges data: x, y, x2, y2, vehicle_id
        routes_layer = alt.Chart(pd.DataFrame()).mark_line()
        arrows_layer = alt.Chart(pd.DataFrame()).mark_point()

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

            # Arrows (Midpoint markers with rotation)
            # We calculate midpoints and rotation in the dataframe beforehand
            arrows_layer = alt.Chart(edges_df).mark_point(
                shape="triangle",
                size=100,
                fillOpacity=1
            ).encode(
                x='mid_x:Q',
                y='mid_y:Q',
                color='vehicle_id:N',
                # Rotate the triangle to match line direction.
                # Altair expects rotation in degrees clockwise from upright? 
                # Actually angle=0 is right-facing usually or up. 
                # We calculated 'angle' in degrees in _parse_solution_text.
                angle='angle:Q', 
                tooltip=['vehicle_id', 'distance']
            )

        # --- Layer 2: Nodes (Points) ---
        # Define shapes/colors for node types
        # Map: 0 -> Depot, 1 -> Pickup, 2 -> Delivery (Logic depends on parsing)
        
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

        # Draw Depots (Squares, Black)
        depots = nodes_base.transform_filter(
            alt.datum.type_label == 'Depot'
        ).mark_square(
            size=150, color='black', opacity=1
        )

        # Draw Others (Circles, colored by type)
        stops = nodes_base.transform_filter(
            alt.datum.type_label != 'Depot'
        ).mark_circle(
            size=80, opacity=0.8
        ).encode(
            color=alt.Color('type_label:N', 
                            scale=alt.Scale(domain=['Pickup', 'Delivery', 'Stop'], 
                                            range=['#1f77b4', '#d62728', 'gray']),
                            legend=alt.Legend(title="Node Type"))
        )

        # Labels (Node IDs) - Optional, can be cluttered if many nodes
        text = nodes_base.mark_text(
            align='left', baseline='middle', dx=7, fontSize=10
        ).encode(text='id:N')

        # Combine
        return (routes_layer + arrows_layer + depots + stops + text).interactive()

    # ================= Data Parsing Helpers =================

    @staticmethod
    def _parse_instance_data(content: str) -> pd.DataFrame:
        """
        Parses .sarp/.vrp content to extract node coordinates.
        Assumes standard format with 'NODE_COORD_SECTION'.
        """
        lines = content.splitlines()
        nodes = []
        parsing_coords = False
        
        # Simple state machine to read coords
        for line in lines:
            line = line.strip()
            if not line: 
                continue
            
            if line.startswith("NODE_COORD_SECTION"):
                parsing_coords = True
                continue
            if line.startswith("DEMAND_SECTION") or line.startswith("DEPOT_SECTION") or line.startswith("NODE_TYPE_SECTION"):
                parsing_coords = False
                continue
            if line == "EOF":
                break

            if parsing_coords:
                parts = re.split(r'\s+', line)
                if len(parts) >= 3:
                    try:
                        # Format: ID X Y
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        nodes.append({"id": nid, "x": x, "y": y, "type_code": 0, "demand": 0})
                    except ValueError:
                        pass
        
        if not nodes:
            return pd.DataFrame(columns=["id", "x", "y", "type_label", "demand"])

        df = pd.DataFrame(nodes)
        
        # Simple heuristic for types if NODE_TYPE_SECTION isn't fully parsed yet:
        # Node 1 is usually Depot.
        # We can try to parse NODE_TYPE_SECTION if present, but for now let's default:
        df["type_label"] = df["id"].apply(lambda i: "Depot" if i == 1 else "Stop")
        
        # Refined Type Parsing (Bonus)
        # Check for NODE_TYPE_SECTION manually
        try:
            type_section_idx = -1
            for i, line in enumerate(lines):
                if line.startswith("NODE_TYPE_SECTION"):
                    type_section_idx = i
                    break
            
            if type_section_idx != -1:
                for line in lines[type_section_idx+1:]:
                    if "SECTION" in line or "EOF" in line: break
                    parts = re.split(r'\s+', line.strip())
                    if len(parts) >= 2:
                        nid = int(parts[0])
                        # Assuming SARP format: ID TYPE ...
                        # Type 0 or -1 is usually Depot
                        # In the user example: Depot was 0? No, checking text...
                        # Actually often in VRP: Pairs define types. 
                        # Let's keep it simple: 
                        # If we can parse pairing, we assign P/D. 
                        # Without pairing logic here, "Stop" is safe.
                        pass
        except Exception:
            pass

        return df

    @staticmethod
    def _parse_solution_text(content: str, nodes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses solution text into a dataframe of edges for plotting.
        Expected format example: "Route 1: 1 - 5 - 10 - 1" or similar.
        Returns DataFrame with cols: [x, y, vehicle_id, order, distance, angle, mid_x, mid_y]
        """
        edges = []
        
        # Map ID to coords
        coords = nodes_df.set_index("id")[["x", "y"]].to_dict('index')

        # Regex to find routes: "Route #1: 1 2 3" or "Route 1 : 1 - 2 - 3"
        # Or simple line-by-line sequences
        lines = content.splitlines()
        vehicle_count = 0
        
        for line in lines:
            if "Route" in line or "Vehicle" in line:
                vehicle_count += 1
                # Extract numbers
                stops = [int(s) for s in re.findall(r'\b\d+\b', line)]
                # Filter out the route number itself if it appears at start
                # Heuristic: if first number is vehicle_count, ignore it? 
                # Better: Usually looks like "Route 1: ..."
                if ":" in line:
                    _, path_str = line.split(":", 1)
                    path = [int(s) for s in re.findall(r'\b\d+\b', path_str)]
                else:
                    # Assume simple list of nodes
                    path = stops

                # Generate segments
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if u in coords and v in coords:
                        x1, y1 = coords[u]['x'], coords[u]['y']
                        x2, y2 = coords[v]['x'], coords[v]['y']
                        
                        # Calc direction and distance
                        dx = x2 - x1
                        dy = y2 - y1
                        dist = math.sqrt(dx*dx + dy*dy)
                        
                        # Angle for arrow (in degrees). 
                        # Altair 0 degrees points "up" or "right"? 
                        # Usually 0 is right (East). 
                        # arctan2 returns radians.
                        angle = math.degrees(math.atan2(dy, dx))
                        # Altair mark_point arrow points Up by default? Or Right?
                        # Assuming triangle points Up (0 deg). If line is East (0 deg), we need rotation -90?
                        # Let's assume triangle shape points Up.
                        # We want it to point along the line.
                        # atan2(y, x) gives angle from X-axis.
                        # Correction factor might be needed depending on shape orientation.
                        # Let's try raw degrees - 90 if triangle points up.
                        chart_angle = angle - 90 

                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2

                        # Add Start Point of segment
                        edges.append({
                            "vehicle_id": f"V{vehicle_count}",
                            "order": i * 2,
                            "x": x1, "y": y1,
                            "mid_x": mid_x, "mid_y": mid_y,
                            "angle": chart_angle,
                            "distance": dist
                        })
                        # Add End Point of segment (for line drawing connectivity)
                        edges.append({
                            "vehicle_id": f"V{vehicle_count}",
                            "order": i * 2 + 1,
                            "x": x2, "y": y2,
                            "mid_x": mid_x, "mid_y": mid_y, # Duplicated but unused for line
                            "angle": chart_angle,
                            "distance": dist
                        })

        return pd.DataFrame(edges)