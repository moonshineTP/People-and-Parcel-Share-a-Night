from typing import List, Tuple

def visualize_instance(
    coords: List[Tuple[float, float]], N: int, M: int, K: int
):
    """Visualize instance with matplotlib, if available."""
    try:
        import matplotlib.pyplot as plt

        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]

        plt.figure(figsize=(6, 6))
        plt.scatter(xs[1:], ys[1:], label="Nodes")
        plt.scatter(xs[0], ys[0], c="red", label="Depot")

        for idx, (x, y) in enumerate(coords):
            plt.text(x, y, str(idx), fontsize=8)

        plt.title(f"Instance coords N={N}, M={M}, K={K}")
        plt.legend()
        plt.show()

    except Exception as e:
        print("Visualization failed (matplotlib not available):", e)