"""
Main application shell for Share-a-Ride problem manager.
Provides top-level navigation between main views using Streamlit.
Run this file to start the app:
    streamlit run share_a_ride/app/app.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

from share_a_ride.app.dashboard import ShareARideDashboard
from share_a_ride.app.visualizer import ShareARideSolutionVisualizer


class ShareARideApp:
    """Main application shell with top-level navigation."""

    def __init__(self) -> None:
        # Page config must be the first Streamlit command
        st.set_page_config(
            page_title="Share-a-Ride Manager",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.dashboard = ShareARideDashboard()
        self.visualizer = ShareARideSolutionVisualizer()

    def _render_sidebar(self) -> str:
        """Render the sidebar navigation."""
        st.sidebar.title("Navigation")
        return st.sidebar.radio("Go to", ["Dashboard", "Visualizer"])

    def _render_visualizer_view(self):
        """Render the visualizer view with inputs."""
        st.header("Solution Visualizer")
        st.write("Paste your Instance (SARP format) and Solution content below to visualize.")

        col1, col2 = st.columns(2)
        with col1:
            instance_input = st.text_area("Instance Content", height=300, help="Paste content of .sarp file here")
        with col2:
            solution_input = st.text_area("Solution Content", height=300, help="Paste solution route text here")

        if st.button("Visualize", type="primary"):
            if instance_input:
                self.visualizer.render(instance_input, solution_input)
            else:
                st.error("Instance content is required.")

    def run(self) -> None:
        """Launch the main application loop."""
        selected_view = self._render_sidebar()

        if selected_view == "Dashboard":
            # The dashboard class handles its own title and layout, 
            # but we can wrap it if needed.
            # Note: Dashboard.run() sets page config too, but Streamlit ignores subsequent calls.
            # We call the internal run method of the dashboard.
            self.dashboard.run()
            
        elif selected_view == "Visualizer":
            self._render_visualizer_view()


if __name__ == "__main__":
    app = ShareARideApp()
    app.run()
