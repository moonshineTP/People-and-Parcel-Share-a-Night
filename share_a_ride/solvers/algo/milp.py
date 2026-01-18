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
    N = problem.N
    M = problem.M
    K = problem.K
    Q = problem.Q
    q_parcels = problem.q
    
    # Convert distance matrix to numpy array for proper 2D indexing
    D_original = np.array(problem.D, dtype=float)
    
    # Transformed node count: 1 (depot) + N (passengers merged) + M (parcel pickups) + M (parcel dropoffs) + 1 (end depot)
    num_nodes_transformed = N + 2 * M + 2
    
    # Define node indices in transformed space
    # V_0 = {0, N+2M+1} (depot at start and end)
    # V_p = {1, ..., N} (passenger pickups, implicitly with drops)
    # V_l = {N+1, ..., N+M} (parcel pickups)
    # V'_l = {N+M+1, ..., N+2M} (parcel dropoffs)
    V_p_indices = list(range(1, N + 1))
    V_l_indices = list(range(N + 1, N + M + 1))
    V_lp_indices = list(range(N + M + 1, N + 2 * M + 1))
    
    # Original node indices mapping
    # Original: 0=depot, 1..N=pass_pickup, N+1..N+M=parcel_pickup, N+M+1..2N+M=pass_drop, 2N+M+1..2N+2M=parcel_drop, 2N+2M=end_depot
    pass_pickup_original = list(range(1, N + 1))
    pass_drop_original = list(range(N + M + 1, 2 * N + M + 1))
    
    # Build transformed distance matrix
    D_transformed = np.zeros((num_nodes_transformed, num_nodes_transformed))
    
    for i in range(num_nodes_transformed):
        for j in range(num_nodes_transformed):
            # Depot to/from any node: use original indices
            if i == 0:
                D_transformed[i, j] = D_original[0, j]
            elif j == 0:
                D_transformed[i, j] = D_original[i, 0]
            elif i == num_nodes_transformed - 1:
                D_transformed[i, j] = D_original[2 * N + 2 * M, j]
            elif j == num_nodes_transformed - 1:
                D_transformed[i, j] = D_original[i, 2 * N + 2 * M]
            else:
                # Both i and j are in V_p ∪ V_l ∪ V'_l
                i_is_pass = i in V_p_indices
                j_is_pass = j in V_p_indices
                
                if not i_is_pass and not j_is_pass:
                    # Both are parcels: direct distance
                    D_transformed[i, j] = D_original[i, j]
                elif i_is_pass and not j_is_pass:
                    # From passenger node i to parcel node j: from i's drop to j
                    i_orig_drop = pass_drop_original[i - 1]
                    D_transformed[i, j] = D_original[i_orig_drop, j]
                elif not i_is_pass and j_is_pass:
                    # From parcel node i to passenger node j: i to j's pickup and drop
                    j_orig_pickup = pass_pickup_original[j - 1]
                    j_orig_drop = pass_drop_original[j - 1]
                    D_transformed[i, j] = D_original[i, j_orig_pickup] + D_original[j_orig_pickup, j_orig_drop]
                else:
                    # Both are passengers: from i's drop to j's pickup to drop
                    i_orig_drop = pass_drop_original[i - 1]
                    j_orig_pickup = pass_pickup_original[j - 1]
                    j_orig_drop = pass_drop_original[j - 1]
                    D_transformed[i, j] = D_original[i_orig_drop, j_orig_pickup] + D_original[j_orig_pickup, j_orig_drop]
    
    # Build q array (weight delta per node)
    q = np.zeros(num_nodes_transformed)
    for idx, node in enumerate(V_l_indices):
        q[node] = q_parcels[idx]  # Positive at pickup
    for idx, node in enumerate(V_lp_indices):
        q[node] = -q_parcels[idx]  # Negative at dropoff
    
    # Compute linearization constants
    # M_tau = 2 * max distance, m_tau = min non-zero distance
    max_dist = np.max(D_transformed)
    M_tau = 2.0 * max_dist
    
    min_dist_nonzero = np.min(D_transformed[D_transformed > 0]) if np.any(D_transformed > 0) else 1.0
    m_tau = min_dist_nonzero
    
    M_1 = M_tau + m_tau
    
    # Compute W_ki bounds per vehicle per node
    # W_ki = min{2*Q_k, 2*Q_k + q_i}
    W_ki = np.zeros((K, num_nodes_transformed))
    for k in range(K):
        for i in range(num_nodes_transformed):
            W_ki[k, i] = min(2 * Q[k], 2 * Q[k] + q[i])
    
    return {
        "D_transformed": D_transformed,
        "q": q,
        "M_tau": M_tau,
        "m_tau": m_tau,
        "M_1": M_1,
        "W_ki": W_ki,
        "N": N,
        "M": M,
        "K": K,
        "V_p_indices": V_p_indices,
        "V_l_indices": V_l_indices,
        "V_lp_indices": V_lp_indices,
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

    _ = _preprocess(problem)  # pylint: disable=unused-variable
    # Preprocessing results will be unpacked when implementing Phase 2+

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
    # Phase 3: Decision Variables (Scaffold - not defined yet)
    # ============================================================================

    # pylint: disable=fixme
    # TODO: Define decision variables
    #   - X[i,j,k]: binary, vehicle k travels from node i to node j
    #   - tau[k,i]: continuous, timestamp of vehicle k at node i
    #   - w[k,i]: continuous, parcel load on vehicle k after node i
    #   - z: continuous, maximum route cost

    # ============================================================================
    # Phase 4: Objective Function
    # ============================================================================

    # pylint: disable=fixme
    # TODO: Define constraints and set objective to minimize z
    # For now, we create a placeholder variable and objective
    z = model.addVar(name="z", lb=0.0)
    model.setObjective(z, GRB.MINIMIZE)

    # ============================================================================
    # Phase 5: Constraints (Not yet implemented)
    # ============================================================================

    # pylint: disable=fixme
    # TODO: Add constraints:
    #   1. Coverage constraints (each request served exactly once)
    #   2. Parcel pairing constraints (pickup and drop for each parcel)
    #   3. Vehicle flow constraints (start/end at depot, flow conservation)
    #   4. Timestamp constraints (subtour elimination via MTZ)
    #   5. Parcel load constraints (capacity and flow conservation)
    #   6. Parcel ordering constraint (pickup before drop)
    #   7. Max cost definition constraint (z >= cost per vehicle)

    # ============================================================================
    # Phase 6: Optimize
    # ============================================================================

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
    # Phase 8: Return Temporary Empty Solution (Not yet parsed from model)
    # ============================================================================

    # pylint: disable=fixme
    # TODO: Parse X variables and reconstruct routes into Solution object
    # For now, return empty solution
    solution = Solution(
        problem=problem,
        routes=[[] for _ in range(problem.K)],
        route_costs=[0] * problem.K,
    )

    return solution, info_dict
