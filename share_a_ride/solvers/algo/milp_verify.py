"""
Verification tests for MILP preprocessing (_preprocess function).

Run with: python milp_verify.py
"""

import sys
from pathlib import Path

import numpy as np

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.solvers.algo.milp import _preprocess, milp


# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# Test 1: Dimensions and node indices
# ============================================================================


def test_dimensions():
    """Verify output dimensions match problem size."""
    N, M, K = 2, 3, 2

    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=[1, 2, 3],
        vehicle_caps=[10, 15],
        dist=[[1] * (2 * N + 2 * M + 2) for _ in range(2 * N + 2 * M + 2)],
    )

    result = _preprocess(problem)

    num_nodes = N + 2 * M + 2

    assert result["D_transformed"].shape == (num_nodes, num_nodes), (
        "D_transformed shape mismatch"
    )
    assert result["q"].shape == (num_nodes,), "q shape mismatch"
    assert result["W_ki"].shape == (K, num_nodes), "W_ki shape mismatch"
    assert result["N"] == N, "N mismatch"
    assert result["M"] == M, "M mismatch"
    assert result["K"] == K, "K mismatch"
    assert len(result["V_p_indices"]) == N, "V_p_indices length mismatch"
    assert len(result["V_l_indices"]) == M, "V_l_indices length mismatch"
    assert len(result["V_lp_indices"]) == M, "V_lp_indices length mismatch"

    print("✓ test_dimensions passed")


# ============================================================================
# Test 2: q array (weight delta)
# ============================================================================


def test_q_array():
    """Verify q array has correct signs and values."""
    N, M, K = 1, 2, 1
    q_input = [3, 5]

    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=q_input,
        vehicle_caps=[20],
        dist=[[1] * (2 * N + 2 * M + 2) for _ in range(2 * N + 2 * M + 2)],
    )

    result = _preprocess(problem)
    q = result["q"]
    V_l = result["V_l_indices"]
    V_lp = result["V_lp_indices"]

    # q should sum to 0 (balance)
    assert np.isclose(np.sum(q), 0), f"q array sum should be 0, got {np.sum(q)}"

    # Parcel pickups should have positive weights
    for idx, node_idx in enumerate(V_l):
        assert q[node_idx] == q_input[idx], (
            f"q[{node_idx}] should be {q_input[idx]}, got {q[node_idx]}"
        )

    # Parcel dropoffs should have negative weights
    for idx, node_idx in enumerate(V_lp):
        assert q[node_idx] == -q_input[idx], (
            f"q[{node_idx}] should be -{q_input[idx]}, got {q[node_idx]}"
        )

    # Non-parcel nodes should have zero weight
    for i in range(len(q)):
        if i not in V_l and i not in V_lp:
            assert q[i] == 0, f"q[{i}] should be 0 for non-parcel node, got {q[i]}"

    print("✓ test_q_array passed")


# ============================================================================
# Test 3: Linearization constants
# ============================================================================


def test_linearization_constants():
    """Verify M_tau, m_tau, M_1 computation."""
    D = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=int)

    dist_matrix = np.vstack(
        [
            np.hstack([D, [[100], [100], [100]]]),
            [[100, 100, 100, 0]],
        ]
    ).tolist()

    problem = ShareARideProblem(
        N=1, M=0, K=1, parcel_qty=[], vehicle_caps=[10], dist=dist_matrix
    )

    result = _preprocess(problem)
    D_t = result["D_transformed"]

    M_tau = result["M_tau"]
    m_tau = result["m_tau"]
    M_1 = result["M_1"]

    max_dist = np.max(D_t)
    nonzero_dists = D_t[D_t > 1e-10]
    min_dist = np.min(nonzero_dists) if len(nonzero_dists) > 0 else 1.0

    assert np.isclose(M_tau, 2.0 * max_dist), "M_tau should be 2*max_dist"
    assert np.isclose(m_tau, min_dist), "m_tau should be min(nonzero distances)"
    assert np.isclose(M_1, M_tau + m_tau), "M_1 should be M_tau + m_tau"

    print("✓ test_linearization_constants passed")


# ============================================================================
# Test 4: W_ki bounds
# ============================================================================


def test_w_ki_bounds():
    """Verify W_ki = min(2*Q_k, 2*Q_k + q_i)."""
    N, M, K = 1, 1, 2
    Q = [10, 20]
    q_input = [3]

    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=q_input,
        vehicle_caps=Q,
        dist=[[1] * (2 * N + 2 * M + 2) for _ in range(2 * N + 2 * M + 2)],
    )

    result = _preprocess(problem)
    W_ki = result["W_ki"]
    q = result["q"]

    for k in range(K):
        for i in range(len(q)):
            expected = min(2 * Q[k], 2 * Q[k] + q[i])
            assert np.isclose(W_ki[k, i], expected), (
                f"W_ki[{k},{i}] should be {expected}, got {W_ki[k, i]}"
            )

    print("✓ test_w_ki_bounds passed")


# ============================================================================
# Test 5: Node index consistency
# ============================================================================


def test_node_indices():
    """Verify node index ranges are correct."""
    N, M, K = 3, 2, 1

    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=[1, 2],
        vehicle_caps=[15],
        dist=[[1] * (2 * N + 2 * M + 2) for _ in range(2 * N + 2 * M + 2)],
    )

    result = _preprocess(problem)

    V_p = result["V_p_indices"]
    V_l = result["V_l_indices"]
    V_lp = result["V_lp_indices"]

    # Check ranges
    assert V_p == list(range(1, N + 1)), (
        f"V_p should be {list(range(1, N + 1))}, got {V_p}"
    )
    assert V_l == list(range(N + 1, N + M + 1)), (
        f"V_l should be {list(range(N + 1, N + M + 1))}, got {V_l}"
    )
    assert V_lp == list(range(N + M + 1, N + 2 * M + 1)), (
        f"V_lp should be {list(range(N + M + 1, N + 2 * M + 1))}, got {V_lp}"
    )

    # Check no overlaps
    all_indices = set(V_p + V_l + V_lp)
    assert len(all_indices) == N + 2 * M, "Node indices should not overlap"

    print("✓ test_node_indices passed")


# ============================================================================
# Test 6: Solution Extraction (end-to-end MILP solver)
# ============================================================================


def test_solution_extraction():
    """Test solution extraction from solved Gurobi model with passenger decompression."""
    # Create minimal problem: 1 passenger, 0 parcels, 1 vehicle
    N, M, K = 1, 0, 1

    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=[],
        vehicle_caps=[100],
        dist=[
            [0, 10, 10],  # Start depot -> pass. 0, pass. 0 drop
            [10, 0, 10],  # Pass. 0 -> pass. 0 drop
            [10, 10, 0],  # Pass. 0 drop -> anywhere
        ],
    )

    # Solve with Gurobi
    solution, info_dict = milp(problem, time_limit=30.0, verbose=True)

    # Verify solution exists
    assert solution is not None, "No solution found"

    # Verify routes exist and are non-empty
    assert len(solution.routes) == K, f"Expected {K} routes, got {len(solution.routes)}"

    # First route should start at 0 (start depot)
    if len(solution.routes[0]) > 0:
        assert solution.routes[0][0] == 0, "Route must start at depot 0"

    # First route should end at 0 (same depot, both start and end)
    if len(solution.routes[0]) > 1:
        assert solution.routes[0][-1] == 0, (
            f"Route must end at depot 0. Got {solution.routes[0][-1]}"
        )

    # Verify route costs are computed
    assert len(solution.route_costs) == K, "Route costs mismatch"
    assert all(cost >= 0 for cost in solution.route_costs), "Negative costs found"

    # Verify solver statistics
    assert "elapsed_time" in info_dict, "Missing elapsed_time"
    assert "status" in info_dict, "Missing status"
    assert info_dict["solver"] == "Gurobi MILP", "Wrong solver name"

    print("✓ test_solution_extraction passed")


def test_solution_extraction_with_parcels():
    """Test solution extraction with both passengers and parcels, verify validity."""
    # Create problem: 2 passengers, 1 parcel, 1 vehicle
    N, M, K = 2, 1, 1

    # Distance matrix: 2*N + 2*M + 1 = 8 nodes
    # 0: depot, 1-2: passenger pickups, 3: parcel pickup, 4-5: passenger drops, 6: parcel drop
    problem = ShareARideProblem(
        N=N,
        M=M,
        K=K,
        parcel_qty=[5],  # 1 parcel with weight 5
        vehicle_caps=[20],  # Vehicle capacity 20
        dist=[
            [0, 5, 7, 3, 5, 7, 3, 0],  # 0: depot
            [5, 0, 4, 6, 2, 6, 6, 5],  # 1: pass 0 pick
            [7, 4, 0, 5, 6, 2, 5, 7],  # 2: pass 1 pick
            [3, 6, 5, 0, 7, 5, 2, 3],  # 3: parcel pick
            [5, 2, 6, 7, 0, 4, 7, 5],  # 4: pass 0 drop
            [7, 6, 2, 5, 4, 0, 5, 7],  # 5: pass 1 drop
            [3, 6, 5, 2, 7, 5, 0, 3],  # 6: parcel drop
            [0, 5, 7, 3, 5, 7, 3, 0],  # 7: end depot (should be same as 0)
        ],
    )

    # Solve with Gurobi
    solution, info_dict = milp(problem, time_limit=30.0, verbose=True)

    # Verify solution exists
    assert solution is not None, "No solution found"

    # Verify solution is valid
    assert solution.is_valid(True), (
        f"Solution is invalid.\nRoutes: {solution.routes}\nCosts: {solution.route_costs}"
    )

    # Verify routes exist
    assert len(solution.routes) == K, f"Expected {K} routes, got {len(solution.routes)}"

    # Verify route costs are computed
    assert len(solution.route_costs) == K, "Route costs mismatch"
    assert all(cost >= 0 for cost in solution.route_costs), "Negative costs found"

    # Verify solver statistics
    assert "elapsed_time" in info_dict, "Missing elapsed_time"
    assert "status" in info_dict, "Missing status"
    assert info_dict["solver"] == "Gurobi MILP", "Wrong solver name"

    print("✓ test_solution_extraction_with_parcels passed")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_dimensions,
        test_q_array,
        test_linearization_constants,
        test_w_ki_bounds,
        test_node_indices,
        test_solution_extraction,
        test_solution_extraction_with_parcels,
    ]

    print("=" * 60)
    print("MILP PREPROCESSING (_preprocess) VERIFICATION")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED:")
            print(f"  {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
