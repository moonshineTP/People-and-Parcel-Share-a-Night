"""
Main application shell for Share-a-Ride problem manager.
Provides top-level navigation between main views using Streamlit.
Run this file to start the app:
    streamlit run share_a_ride/app/app.py
"""

import sys
from pathlib import Path

# Add root path to sys.path for module imports
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

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


    def run(self) -> None:
        """Launch the main application loop."""
        selected_view = self._render_sidebar()

        if selected_view == "Dashboard":
            self.dashboard.run()
        elif selected_view == "Visualizer":
            self.visualizer.run()



if __name__ == "__main__":
    app = ShareARideApp()
    app.run()
