"""
Streamlit solution visualizer for the Share-a-Ride app.
"""
from __future__ import annotations

import math
import re
import pandas as pd
import altair as alt
import streamlit as st

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
            nodes_df, depot_ids = self._parse_instance_data(instance_content)
        except Exception as e:
            st.error(f"Failed to parse instance data: {e}")
            return

        routes_df = pd.DataFrame()
        reported_cost = None
        
        if solution_content:
            try:
                # Use the first depot found as the default start/end for routes
                default_depot = depot_ids[0] if depot_ids else 1
                routes_df, reported_cost = self._parse_solution_text(solution_content, nodes_df, default_depot)
            except Exception as e:
                st.warning(f"Could not parse solution fully: {e}")

        # 2. Draw Chart
        st.markdown("### Solution Visualization")
        
        if not nodes_df.empty:
            chart = self._build_altair_chart(nodes_df, routes_df)
            st.altair_chart(chart, width='stretch')
        
        # 3. Metrics (if solution exists)
        if not routes_df.empty:
            self._render_metrics(routes_df, reported_cost)

    def _render_metrics(self, routes_edge_df: pd.DataFrame, reported_cost: float | None):
        """Display summary metrics based on the calculated edges."""
        # Calculate total distance based on Euclidean distance of segments
        total_dist = routes_edge_df["distance"].sum()
        num_vehicles = routes_edge_df["vehicle_id"].nunique()
        
        cols = st.columns(3)
        cols[0].metric("Calculated Distance (Euclidean)", f"{total_dist:,.2f}")
        cols[1].metric("Vehicles Used", f"{num_vehicles}")
        if reported_cost is not None:
            cols[2].metric("Reported Cost (from file)", f"{reported_cost:,.2f}")

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
            arrows_layer = alt.Chart(edges_df).mark_point(
                shape="triangle",
                size=100,
                fillOpacity=1
            ).encode(
                x='mid_x:Q',
                y='mid_y:Q',
                color='vehicle_id:N',
                angle='angle:Q', 
                tooltip=['vehicle_id', 'distance']
            )

        # --- Layer 2: Nodes (Points) ---
        
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

        # Draw Pickups (Up triangles)
        pickups = nodes_base.transform_filter(
            alt.datum.type_label == 'Pickup'
        ).mark_point(
            shape="triangle-up", size=100, filled=True, opacity=0.8
        ).encode(
            color=alt.value("#1f77b4") # Blue
        )

        # Draw Deliveries (Down triangles)
        deliveries = nodes_base.transform_filter(
            alt.datum.type_label == 'Delivery'
        ).mark_point(
            shape="triangle-down", size=100, filled=True, opacity=0.8
        ).encode(
            color=alt.value("#d62728") # Red
        )

        # Draw Others/Stops (Circles, Gray)
        stops = nodes_base.transform_filter(
            (alt.datum.type_label != 'Depot') &
            (alt.datum.type_label != 'Pickup') &
            (alt.datum.type_label != 'Delivery')
        ).mark_circle(
            size=80, opacity=0.6
        ).encode(
            color=alt.value("gray")
        )

        # Labels (Node IDs) - Optional, can be cluttered if many nodes
        # Only show for small instances or on hover? Let's show small text always.
        text = nodes_base.mark_text(
            align='left', baseline='middle', dx=7, fontSize=10
        ).encode(text='id:N')

        # Combine
        return (routes_layer + arrows_layer + depots + pickups + deliveries + stops + text).interactive()

    # ================= Data Parsing Helpers =================

    @staticmethod
    def _parse_instance_data(content: str) -> tuple[pd.DataFrame, list[int]]:
        """
        Parses .sarp/.vrp content to extract node info.
        Returns:
            - DataFrame with cols: [id, x, y, demand, type_label]
            - List of depot IDs
        """
        lines = content.splitlines()
        
        coords = {}  # {id: (x, y)}
        demands = {} # {id: demand}
        node_types = {} # {id: type_code}
        depots = []

        current_section = None

        for line in lines:
            line = line.strip()
            if not line: continue
            
            if line.startswith("NODE_COORD_SECTION"):
                current_section = "COORD"
                continue
            elif line.startswith("DEMAND_SECTION"):
                current_section = "DEMAND"
                continue
            elif line.startswith("DEPOT_SECTION"):
                current_section = "DEPOT"
                continue
            elif line.startswith("NODE_TYPE_SECTION"):
                current_section = "TYPE"
                continue
            elif "SECTION" in line:
                current_section = None # Other sections we ignore for now
                continue
            elif line == "EOF":
                break

            parts = re.split(r'\s+', line)
            
            if current_section == "COORD" and len(parts) >= 3:
                try:
                    coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
                except ValueError: pass
            
            elif current_section == "DEMAND" and len(parts) >= 2:
                try:
                    demands[int(parts[0])] = float(parts[1])
                except ValueError: pass
            
            elif current_section == "DEPOT":
                try:
                    val = int(parts[0])
                    if val != -1:
                        depots.append(val)
                except ValueError: pass

            elif current_section == "TYPE" and len(parts) >= 3:
                # Format likely: ID ID TYPE? Or ID TYPE ...?
                # The user example showed "122 122 3". Let's assume ID is first col, Type is 3rd?
                # Actually commonly in SARP: ID OriginalID Type
                # Let's try to parse the last numeric column as type if 3 cols, or 2nd if 2 cols.
                try:
                    nid = int(parts[0])
                    ntype = int(parts[-1]) 
                    node_types[nid] = ntype
                except ValueError: pass

        # Build DataFrame
        data = []
        for nid, (x, y) in coords.items():
            d = demands.get(nid, 0)
            
            # Determine Label
            # Heuristic map based on common VRP/SARP: 
            # 0: Depot, 1: Station/Stop, 2: Pickup, 3: Delivery?
            # Or -1: Depot. 
            # We use the depots list as truth for Depot.
            
            t_code = node_types.get(nid, -1)
            
            if nid in depots:
                t_label = "Depot"
            else:
                if t_code == 2: t_label = "Pickup"
                elif t_code == 3: t_label = "Delivery"
                else: t_label = "Stop"
            
            data.append({
                "id": nid,
                "x": x,
                "y": y,
                "demand": d,
                "type_label": t_label
            })

        df = pd.DataFrame(data)
        return df, depots

    @staticmethod
    def _parse_solution_text(content: str, nodes_df: pd.DataFrame, default_depot: int) -> tuple[pd.DataFrame, float | None]:
        """
        Parses solution text into a dataframe of edges.
        Also parses the 'Cost' line.
        """
        edges = []
        coords = nodes_df.set_index("id")[["x", "y"]].to_dict('index')
        lines = content.splitlines()
        
        reported_cost = None
        
        for line in lines:
            line = line.strip()
            if not line: continue

            # Parse Cost
            if line.lower().startswith("cost"):
                try:
                    # "Cost 5623.47"
                    parts = line.split()
                    if len(parts) >= 2:
                        reported_cost = float(parts[1])
                except ValueError: pass
                continue

            # Parse Routes
            if "Route" in line or "Vehicle" in line:
                # Extract route ID e.g. Route #1 -> 1
                route_match = re.search(r'#(\d+)', line)
                vid = f"V{route_match.group(1)}" if route_match else "V?"
                
                # Extract stops
                # Look for part after colon
                if ":" in line:
                    path_str = line.split(":", 1)[1]
                else:
                    path_str = line
                
                # Get all integers
                path = [int(s) for s in re.findall(r'\b\d+\b', path_str)]
                
                if not path: continue

                # Inject Depot if needed
                # If first node is not depot, prepend it
                if path[0] != default_depot:
                    path.insert(0, default_depot)
                # If last node is not depot, append it
                if path[-1] != default_depot:
                    path.append(default_depot)

                # Generate segments
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if u in coords and v in coords:
                        x1, y1 = coords[u]['x'], coords[u]['y']
                        x2, y2 = coords[v]['x'], coords[v]['y']
                        
                        dx, dy = x2 - x1, y2 - y1
                        dist = math.sqrt(dx*dx + dy*dy)
                        angle = math.degrees(math.atan2(dy, dx)) - 90 
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

                        edges.append({
                            "vehicle_id": vid,
                            "order": i * 2,
                            "x": x1, "y": y1,
                            "mid_x": mid_x, "mid_y": mid_y,
                            "angle": angle,
                            "distance": dist
                        })
                        edges.append({
                            "vehicle_id": vid,
                            "order": i * 2 + 1,
                            "x": x2, "y": y2,
                            "mid_x": mid_x, "mid_y": mid_y,
                            "angle": angle,
                            "distance": dist
                        })

        return pd.DataFrame(edges), reported_cost