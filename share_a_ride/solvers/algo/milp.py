"""
Mixed Integer Linear Programming (MILP) solver for the Share-a-Ride Problem using Gurobi.

This module provides an exact optimization solver using Gurobi's branch-and-cut algorithm.
"""

from typing import Dict, Any, Optional, Tuple
import time
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "Gurobi is not installed. Install it separately: pip install gurobipy"
    ) from exc

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution


def _extract_routes_from_model(x: Dict, num_nodes: int, k: int, n: int, m: int) -> list:
    """
    Extract routes from solved model X variables.

    For each vehicle, follows X arcs from node 0 (start depot) to node num_nodes-1
    (end depot) in the transformed space, then decompresses passenger nodes to
    original space.

    Args:
        x: Dictionary of X variables indexed by (i, j, k).
        num_nodes: Total nodes in transformed space (N + 2M + 2).
        k: Number of vehicles.
        n: Number of passengers (for decompression).
        m: Number of parcels (for decompression).

    Returns:
        List of routes, one per vehicle. Each route is a list of node indices in
        the original space (accounting for passenger node decompression).
    """
    routes = []
    end_depot = num_nodes - 1

    for k_idx in range(k):
        route_transformed = [0]  # Start at depot 0
        current = 0

        # Build route in transformed space by following X arcs
        while current != end_depot:
            # Find next node
            next_node = None
            for j in range(num_nodes):
                if x[current, j, k_idx].X > 0.5:  # X variable is binary, ~1 if arc used
                    next_node = j
                    break

            if next_node is None:
                break  # Route incomplete

            route_transformed.append(next_node)
            current = next_node

        # Decompress passenger nodes: V_p node i becomes [i, N+M+i] in original space
        route_original = []
        for node_t in route_transformed:
            if node_t == 0:
                # Start depot
                route_original.append(0)
            elif node_t == end_depot:
                # End depot in transformed space maps to node 0 in original space
                route_original.append(0)
            elif 1 <= node_t <= n:
                # Passenger node in V_p: decompress to pickup and drop
                # In original space: pickup at node_t, drop at N+M+node_t
                route_original.append(node_t)
                route_original.append(n + m + node_t)
            else:
                # Parcel pickup or drop node: keep as-is
                route_original.append(node_t)

        routes.append(route_original)

    return routes


def _calculate_route_costs(routes: list, distance_matrix: list) -> list:
    """
    Calculate total cost for each route using original distance matrix.

    Args:
        routes: List of routes (node sequences in original space).
        distance_matrix: Original problem distance matrix (2D list/array).

    Returns:
        List of route costs, one per vehicle.
    """
    d_orig = np.array(distance_matrix, dtype=float)
    route_costs = []

    for route in routes:
        cost = 0.0
        for i in range(len(route) - 1):
            cost += d_orig[route[i], route[i + 1]]
        route_costs.append(cost)

    return route_costs


def _preprocess(problem: ShareARideProblem) -> Dict[str, Any]:
    """
    Preprocess the problem for MILP formulation.

    Transforms the problem by merging passenger pickup+drop nodes and computes:
    - Transformed distance matrix
    - Weight delta array (q_i)
    - Linearization constants (M_tau, m_tau, M_1)
    - Weight bounds per vehicle per node (W_ki)

    Returns:
        Dictionary with preprocessed data:
            - D_transformed: Transformed distance matrix (shape: (N+2M+1, N+2M+1))
            - q: Weight delta per node (shape: (N+2M+1,))
            - M_tau: Maximum timestamp (2 * max distance)
            - m_tau: Minimum timestamp (min non-zero distance)
            - M_1: Linearization constant (M_tau + m_tau)
            - W_ki: Weight bounds per vehicle per node (shape: (K, N+2M+1))
            - N: Number of passengers
            - M: Number of parcels
            - K: Number of vehicles
            - V_p_indices: Indices of passenger nodes in transformed space
            - V_l_indices: Indices of parcel pickup nodes in transformed space
            - V_lp_indices: Indices of parcel drop nodes in transformed space
    """
    n = problem.N
    m = problem.M
    k = problem.K
    vehicle_caps = problem.Q
    q_parcels = problem.q

    # Convert distance matrix to numpy array for proper 2D indexing
    d_original = np.array(problem.D, dtype=float)

    # Transformed node count:
    # 1 (depot) + N (passengers merged) + M (parcel pickups) + M (parcel dropoffs) + 1 (end depot)
    num_nodes_transformed = n + 2 * m + 2

    # Define node indices in transformed space
    # V_0 = {0, N+2M+1} (depot at start and end)
    # V_p = {1, ..., N} (passenger pickups, implicitly with drops)
    # V_l = {N+1, ..., N+M} (parcel pickups)
    # V'_l = {N+M+1, ..., N+2M} (parcel dropoffs)
    v_p_indices = list(range(1, n + 1))
    v_l_indices = list(range(n + 1, n + m + 1))
    v_lp_indices = list(range(n + m + 1, n + 2 * m + 1))

    # Original node indices mapping
    # Original: 0=depot, 1..N=pass_pickup,
    # N+1..N+M=parcel_pickup, N+M+1..2N+M=pass_drop, 2N+M+1..2N+2M=parcel_drop, 2N+2M=end_depot
    pass_pickup_original = list(range(1, n + 1))
    pass_drop_original = list(range(n + m + 1, 2 * n + m + 1))

    # Build transformed distance matrix
    d_transformed = np.zeros((num_nodes_transformed, num_nodes_transformed))

    for i in range(num_nodes_transformed):
        for j in range(num_nodes_transformed):
            # Depot to/from any node: use original indices
            if i == 0:
                d_transformed[i, j] = d_original[0, j]
            elif j == 0:
                d_transformed[i, j] = d_original[i, 0]
            elif i == num_nodes_transformed - 1:
                d_transformed[i, j] = d_original[2 * n + 2 * m, j]
            elif j == num_nodes_transformed - 1:
                d_transformed[i, j] = d_original[i, 2 * n + 2 * m]
            else:
                # Both i and j are in V_p ∪ V_l ∪ V'_l
                i_is_pass = i in v_p_indices
                j_is_pass = j in v_p_indices

                if not i_is_pass and not j_is_pass:
                    # Both are parcels: direct distance
                    d_transformed[i, j] = d_original[i, j]
                elif i_is_pass and not j_is_pass:
                    # From passenger node i to parcel node j: from i's drop to j
                    i_orig_drop = pass_drop_original[i - 1]
                    d_transformed[i, j] = d_original[i_orig_drop, j]
                elif not i_is_pass and j_is_pass:
                    # From parcel node i to passenger node j: i to j's pickup and drop
                    j_orig_pickup = pass_pickup_original[j - 1]
                    j_orig_drop = pass_drop_original[j - 1]
                    d_transformed[i, j] = (
                        d_original[i, j_orig_pickup]
                        + d_original[j_orig_pickup, j_orig_drop]
                    )
                else:
                    # Both are passengers: from i's drop to j's pickup to drop
                    i_orig_drop = pass_drop_original[i - 1]
                    j_orig_pickup = pass_pickup_original[j - 1]
                    j_orig_drop = pass_drop_original[j - 1]
                    d_transformed[i, j] = (
                        d_original[i_orig_drop, j_orig_pickup]
                        + d_original[j_orig_pickup, j_orig_drop]
                    )

    # Build q array (weight delta per node)
    q_delta = np.zeros(num_nodes_transformed)
    for idx, node in enumerate(v_l_indices):
        q_delta[node] = q_parcels[idx]  # Positive at pickup
    for idx, node in enumerate(v_lp_indices):
        q_delta[node] = -q_parcels[idx]  # Negative at dropoff

    # Compute linearization constants
    # M_tau = 2 * max distance, m_tau = min non-zero distance
    max_dist = np.max(d_transformed)
    max_tau = 2.0 * max_dist

    min_dist_nonzero = (
        np.min(d_transformed[d_transformed > 0]) if np.any(d_transformed > 0) else 1.0
    )
    m_tau = min_dist_nonzero

    m_1 = max_tau + m_tau

    # Compute W_ki bounds per vehicle per node
    # W_ki = min{2*Q_k, 2*Q_k + q_i}
    w_ki = np.zeros((k, num_nodes_transformed))
    for k_idx in range(k):
        for i in range(num_nodes_transformed):
            w_ki[k_idx, i] = min(
                2 * vehicle_caps[k_idx], 2 * vehicle_caps[k_idx] + q_delta[i]
            )

    return {
        "D_transformed": d_transformed,
        "q": q_delta,
        "M_tau": max_tau,
        "m_tau": m_tau,
        "M_1": m_1,
        "W_ki": w_ki,
        "N": n,
        "M": m,
        "K": k,
        "V_p_indices": v_p_indices,
        "V_l_indices": v_l_indices,
        "V_lp_indices": v_lp_indices,
    }


def milp(
    problem: ShareARideProblem,
    partial: Optional[Any] = None,  # pylint: disable=unused-argument
    time_limit: float = 30.0,
    verbose: bool = False,
    **_kwargs,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Solve the Share-a-Ride Problem using Mixed Integer Linear Programming with Gurobi.

    Args:
        problem: The ShareARideProblem instance to solve.
        partial: Optional partial solution for warm-starting (not yet implemented).
        time_limit: Time limit for the solver in seconds.
        verbose: Whether to print Gurobi output.
        **kwargs: Additional solver parameters.

    Returns:
        A tuple of (solution, info_dict) where:
            - solution: Solution object if feasible, None if timeout/infeasible.
            - info_dict: Dictionary with solver statistics (elapsed_time, status, gap, etc).
    """

    start_time = time.time()
    info_dict = {}

    # ============================================================================
    # Phase 1: Preprocessing
    # ============================================================================

    preproc = _preprocess(problem)
    d_transformed = preproc["D_transformed"]
    max_tau = preproc["M_tau"]
    w_ki = preproc["W_ki"]
    m = preproc["M"]
    n = preproc["N"]
    k = preproc["K"]

    num_nodes = n + 2 * m + 2

    # ============================================================================
    # Phase 2: Model Setup
    # ============================================================================

    # Create Gurobi model
    model = gp.Model("SARP")

    # Configure solver output
    if not verbose:
        model.setParam(GRB.Param.OutputFlag, 0)

    # Set time limit
    model.setParam(GRB.Param.TimeLimit, time_limit)

    # ============================================================================
    # Phase 3: Decision Variables
    # ============================================================================

    # Binary variables: X[i,j,k] = vehicle k travels from node i to node j
    x = {}
    for k_idx in range(k):
        for i in range(num_nodes):
            for j in range(num_nodes):
                x[i, j, k_idx] = model.addVar(
                    vtype=GRB.BINARY, name=f"X_{i}_{j}_{k_idx}"
                )

    # Continuous variables: tau[k,i] = timestamp of vehicle k at node i
    tau = {}
    for k_idx in range(k):
        for i in range(num_nodes):
            tau[k_idx, i] = model.addVar(
                lb=0.0, ub=max_tau, vtype=GRB.CONTINUOUS, name=f"tau_{k_idx}_{i}"
            )

    # Continuous variables: w[k,i] = parcel load on vehicle k after visiting node i
    w = {}
    for k_idx in range(k):
        for i in range(num_nodes):
            w[k_idx, i] = model.addVar(
                lb=0.0, ub=w_ki[k_idx, i], vtype=GRB.CONTINUOUS, name=f"w_{k_idx}_{i}"
            )

    # Continuous variable: z = maximum route cost (objective)
    z = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    # ============================================================================
    # Phase 4: Objective Function
    # ============================================================================

    # Minimize the maximum route cost across all vehicles
    model.setObjective(z, GRB.MINIMIZE)

    # ============================================================================
    # Phase 5: Constraints
    # ============================================================================

    # Formulation Constraint (1): Coverage - each passenger pickup and parcel pickup served exactly once
    for i in preproc["V_p_indices"] + preproc["V_l_indices"]:
        model.addConstr(
            gp.quicksum(x[i, j, ik] for j in range(num_nodes) for ik in range(k)) == 1,
            name=f"cov_{i}",
        )

    # Formulation Constraint (2): Parcel pairing - pickup j and drop j+M paired for each vehicle
    for j_idx, j in enumerate(preproc["V_l_indices"]):
        j_drop = preproc["V_lp_indices"][j_idx]
        for k_idx in range(k):
            model.addConstr(
                gp.quicksum(x[i, j, k_idx] for i in range(num_nodes))
                == gp.quicksum(x[i, j_drop, k_idx] for i in range(num_nodes)),
                name=f"pair_{j}_{j_drop}_{k_idx}",
            )

    # Formulation Constraint (3): Vehicle start and end - each vehicle departs start depot and returns to end depot exactly once
    end_depot = num_nodes - 1
    for k_idx in range(k):
        model.addConstr(
            gp.quicksum(x[0, i, k_idx] for i in range(num_nodes)) == 1,
            name=f"start_{k_idx}",
        )
        model.addConstr(
            gp.quicksum(x[i, end_depot, k_idx] for i in range(num_nodes)) == 1,
            name=f"end_{k_idx}",
        )

    # Formulation Constraint (4): No entry to start depot and no exit from end depot
    for k_idx in range(k):
        model.addConstr(
            gp.quicksum(x[i, 0, k_idx] for i in range(num_nodes)) == 0,
            name=f"no_entry_{k_idx}",
        )
        model.addConstr(
            gp.quicksum(x[end_depot, i, k_idx] for i in range(num_nodes)) == 0,
            name=f"no_exit_{k_idx}",
        )

    # Formulation Constraint (5): Flow conservation - for each non-depot node
    for i in range(1, num_nodes - 1):
        for k_idx in range(k):
            model.addConstr(
                gp.quicksum(x[i, j, k_idx] for j in range(num_nodes))
                == gp.quicksum(x[j, i, k_idx] for j in range(num_nodes)),
                name=f"flow_{i}_{k_idx}",
            )

    # Objective definition: z >= cost of each vehicle's route
    # (Formulation Constraints (6)-(9) for timestamp ordering and weight validity not yet implemented)
    for k_idx in range(k):
        model.addConstr(
            z
            >= gp.quicksum(
                d_transformed[i, j] * x[i, j, k_idx]
                for i in range(num_nodes)
                for j in range(num_nodes)
            ),
            name=f"maxcost_{k_idx}",
        )

    model.optimize()

    elapsed_time = time.time() - start_time

    # ============================================================================
    # Phase 7: Extract Results
    # ============================================================================

    # Populate info dictionary with solver statistics
    info_dict["elapsed_time"] = elapsed_time
    info_dict["solver"] = "Gurobi MILP"
    info_dict["time_limit"] = time_limit

    if model.status == GRB.OPTIMAL:
        info_dict["status"] = "optimal"
    elif model.status == GRB.TIME_LIMIT:
        info_dict["status"] = "time_limit"
    elif model.status == GRB.INFEASIBLE:
        info_dict["status"] = "infeasible"
    else:
        info_dict["status"] = f"status_{model.status}"

    if model.solCount > 0:
        info_dict["objective_value"] = model.objVal
        info_dict["gap"] = model.MIPGap if model.status != GRB.OPTIMAL else 0.0
    else:
        info_dict["objective_value"] = None
        info_dict["gap"] = None

    # ============================================================================
    # Phase 8: Solution Extraction
    # ============================================================================

    if model.solCount > 0:
        # Extract routes from X variables
        routes = _extract_routes_from_model(x, num_nodes, k, n, m)
        # Calculate route costs using original distance matrix
        route_costs = _calculate_route_costs(routes, problem.D)
        solution = Solution(
            problem=problem,
            routes=routes,
            route_costs=route_costs,
        )
    else:
        # No solution found
        solution = None

    return solution, info_dict
