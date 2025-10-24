"""
Main application shell for Share-a-Ride problem manager.
Provides top-level navigation between main views.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from share_a_ride.app.dashboard import ShareARideDashboard
from share_a_ride.app.visualizer import ShareARideSolutionVisualizer

class ShareARideApp:
    """Main application shell with top-level navigation."""

    def __init__(self, master: tk.Tk | None = None) -> None:
        self.root = master or tk.Tk()
        self.root.title("Share-a-Ride Manager")
        self.root.geometry("900x600")

        self.views: dict[str, ttk.Frame] = {}
        self._build_header()
        self._build_views()
        self.show_view("dashboard")


    def _build_header(self) -> None:
        # Build the application header with title and navigation buttons.
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X)

        title = ttk.Label(header, text="Share-a-Ride Problem Manager",
            font=("Segoe UI", 30, "bold")
        )
        title.pack(side=tk.LEFT, padx=16, pady=12)

        nav = ttk.Frame(header)
        nav.pack(side=tk.RIGHT, padx=16)

        buttons = {
            "Dashboard": lambda: self.show_view("dashboard"),
            "Visualizer": lambda: self.show_view("visualizer"),
        }
        for label, action in buttons.items():
            ttk.Button(nav, text=label, command=action) \
                .pack(side=tk.LEFT, padx=4)


    def _build_views(self) -> None:
        """
        Build the main application views: dashboard and visualizer.
        """
        container = ttk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True)

        # Enable a lightweight solver config dashboard to test data loading
        dashboard = ShareARideDashboard(container)
        self.views["dashboard"] = dashboard.frame

        # Keep visualizer placeholder as-is
        visualizer = ShareARideSolutionVisualizer(container)
        self.views["visualizer"] = visualizer.frame


    def _make_placeholder(
            self, parent: tk.Widget, title: str, subtitle: str = ""
        ) -> ttk.Frame:
        """
        Create a simple placeholder frame with a title and optional subtitle.
        """
        frame = ttk.Frame(parent)

        # Title
        ttk.Label(frame, text=title, font=("Segoe UI", 16, "bold")).pack(pady=(24, 8))

        # Subtitle
        if subtitle:
            ttk.Label(frame, text=subtitle, justify=tk.CENTER).pack(pady=(0, 12))

        # Instruction
        ttk.Label(frame, text="Use the navigation to switch views.").pack()

        return frame


    def show_view(self, key: str) -> None:
        """
        Show the specified view by key, hiding others.
        """
        # Forget all past views
        for frame in self.views.values():
            frame.pack_forget()

        # Show the requested view
        view = self.views.get(key)
        if view is None:
            messagebox.showerror("Navigation Error", f"Unknown view: {key}")
            return
        view.pack(fill=tk.BOTH, expand=True)


    def run(self) -> None:
        """
        Launch the main application loop.
        """
        self.root.mainloop()



if __name__ == "__main__":
    ShareARideApp().run()
