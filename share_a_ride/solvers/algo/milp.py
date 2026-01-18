"""
Mixed Integer Linear Programming (MILP) solver for the Share-a-Ride Problem.

This module provides exact optimization using pluggable solver backends via a common model interface.
Currently supports Gurobi backend.
"""

from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
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
from share_a_ride.solvers.algo.Algo import AlgoSolver
from share_a_ride.solvers.algo.utils import exact_problems


# ============================================================================
# Model Interface (Solver-Agnostic Abstraction)
# ============================================================================


class Model(ABC):
    """
    Abstract interface for MILP solver models.

    Wraps solver libraries (Gurobi, Pyomo, etc) and exposes common operations.
    Methods return native solver objects (variables, expressions) directly.
    """

    @abstractmethod
    def createVar(
        self,
        lb: float = 0.0,
        ub: float = float("inf"),
        vtype: str = "C",
        name: str = "",
    ) -> Any:
        """Create a variable. Returns native solver variable."""

    @abstractmethod
    def quicksum(self, expr) -> Any:
        """Sum expression. Returns native solver expression."""

    @abstractmethod
    def addConstr(self, constraint, name: str = "") -> None:
        """Add a constraint to the model."""

    @abstractmethod
    def setObjective(self, objective, sense: str = "min") -> None:
        """Set the objective function."""

    @abstractmethod
    def setParam(self, param: str, value: Any) -> None:
        """Set a solver parameter."""

    @abstractmethod
    def optimize(self) -> None:
        """Solve the model."""

    @abstractmethod
    def update(self) -> None:
        """Update the model state."""

    @abstractmethod
    def getVars(self) -> list:
        """Get all variables in the model."""

    @property
    @abstractmethod
    def status(self) -> int:
        """Get solver status code."""

    @property
    @abstractmethod
    def solCount(self) -> int:
        """Get number of solutions found."""

    @property
    @abstractmethod
    def objVal(self) -> float:
        """Get objective value."""

    @property
    @abstractmethod
    def MIPGap(self) -> float:
        """Get MIP gap."""

    @property
    @abstractmethod
    def NumVars(self) -> int:
        """Get total number of variables."""

    @property
    @abstractmethod
    def NumConstrs(self) -> int:
        """Get total number of constraints."""

    @abstractmethod
    def get_status_code(self, status_name: str) -> int:
        """Get solver-specific status code by name."""


class GurobiModel(Model):
    """Gurobi solver model wrapper implementing the Model interface."""

    def __init__(self, name: str = "Model"):
        self._model = gp.Model(name)

    def createVar(
        self,
        lb: float = 0.0,
        ub: float = float("inf"),
        vtype: str = "C",
        name: str = "",
    ) -> Any:
        """Create a variable and return native Gurobi variable."""
        # Map generic vtype to Gurobi constants
        if vtype == "B":
            gurobi_vtype = GRB.BINARY
        elif vtype == "I":
            gurobi_vtype = GRB.INTEGER
        else:  # "C" for continuous
            gurobi_vtype = GRB.CONTINUOUS

        return self._model.addVar(lb=lb, ub=ub, vtype=gurobi_vtype, name=name)

    def quicksum(self, expr) -> Any:
        """Sum expression using Gurobi's quicksum."""
        return gp.quicksum(expr)

    def addConstr(self, constraint, name: str = "") -> None:
        """Add a constraint to the Gurobi model."""
        self._model.addConstr(constraint, name=name)

    def setObjective(self, objective, sense: str = "min") -> None:
        """Set objective function (min or max)."""
        gurobi_sense = GRB.MINIMIZE if sense == "min" else GRB.MAXIMIZE
        self._model.setObjective(objective, gurobi_sense)

    def setParam(self, param: str, value: Any) -> None:
        """Set a Gurobi parameter by name."""
        # Map common parameter names to Gurobi constants
        param_map = {
            "OutputFlag": GRB.Param.OutputFlag,
            "TimeLimit": GRB.Param.TimeLimit,
        }
        gurobi_param = param_map.get(param)
        if gurobi_param is None:
            raise ValueError(f"Unknown parameter: {param}")
        self._model.setParam(gurobi_param, value)

    def optimize(self) -> None:
        """Solve the model."""
        self._model.optimize()

    def update(self) -> None:
        """Update model state."""
        self._model.update()

    def getVars(self) -> list:
        """Get all variables in the model."""
        return self._model.getVars()

    @property
    def status(self) -> int:
        """Get solver status."""
        return self._model.status

    @property
    def solCount(self) -> int:
        """Get number of solutions found."""
        return self._model.solCount

    @property
    def objVal(self) -> float:
        """Get objective value."""
        return self._model.objVal

    @property
    def MIPGap(self) -> float:
        """Get MIP gap."""
        return self._model.MIPGap

    @property
    def NumVars(self) -> int:
        """Get number of variables."""
        return self._model.NumVars

    @property
    def NumConstrs(self) -> int:
        """Get number of constraints."""
        return self._model.NumConstrs

    def get_status_code(self, status_name: str) -> int:
        """Get Gurobi status code by name."""
        status_map = {
            "OPTIMAL": GRB.OPTIMAL,
            "TIME_LIMIT": GRB.TIME_LIMIT,
            "INFEASIBLE": GRB.INFEASIBLE,
        }
        return status_map.get(status_name, -1)


def _create_model(solver_name: str, name: str = "SARP") -> Model:
    """Factory function to create a solver model wrapper."""
    solver_lower = solver_name.lower()
    if solver_lower == "gurobi":
        return GurobiModel(name)
    else:
        raise ValueError(
            f"Unknown MILP solver: '{solver_name}'. Supported solvers: 'gurobi'"
        )


def _build_arc_caches(n, m):
    num_nodes = n + 2 * m + 2
    arcs = {
        (i, j)
        for i in range(0, num_nodes - 1)
        for j in range(1, num_nodes)
        if i not in (j, j + m)
    }
    nodes_from = {
        i: [j for j in range(num_nodes) if (i, j) in arcs] for i in range(num_nodes)
    }
    nodes_to = {
        j: [i for i in range(num_nodes) if (i, j) in arcs] for j in range(num_nodes)
    }
    return (arcs, nodes_from, nodes_to, num_nodes)


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
    arcs, nodes_from, nodes_to, _ = _build_arc_caches(n, m)

    for k_idx in range(k):
        route_transformed = [0]  # Start at depot 0
        current = 0

        # Build route in transformed space by following X arcs
        while current != end_depot:
            # Find next node
            next_node = None
            for j in nodes_from[current]:
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
                # Parcel pickup or drop node: map to original indices
                if n + 1 <= node_t <= n + m:
                    route_original.append(node_t)
                else:
                    route_original.append(node_t + n)
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
    max_tau = max_dist

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
    solver: str = "gurobi",
    **_kwargs,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Solve the Share-a-Ride Problem using Mixed Integer Linear Programming.

    Supports pluggable solver backends via abstracted model interface.
    Currently supports 'gurobi' and 'pyomo'.

    Args:
        problem: The ShareARideProblem instance to solve.
        partial: Optional partial solution for warm-starting (not yet implemented).
        time_limit: Time limit for the solver in seconds.
        verbose: Whether to print solver output.
        solver: Solver backend to use ("gurobi" or "pyomo").
        **kwargs: Additional solver parameters (unused).

    Returns:
        A tuple of (solution, info_dict) where:
            - solution: Solution object if feasible, None if timeout/infeasible.
            - info_dict: Dictionary with solver statistics (elapsed_time, status, gap, etc).

    Raises:
        ValueError: If solver name is invalid.
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
    m: int = preproc["M"]
    n: int = preproc["N"]
    k: int = preproc["K"]

    arcs, nodes_from, nodes_to, num_nodes = _build_arc_caches(n, m)

    # ============================================================================
    # Phase 2: Model Creation (Solver Selection)
    # ============================================================================

    model = _create_model(solver, name="SARP")

    # Configure solver output
    if not verbose:
        model.setParam("OutputFlag", 0)

    # Set time limit
    model.setParam("TimeLimit", time_limit)

    # ============================================================================
    # Phase 3: Decision Variables
    # ============================================================================

    # Binary variables: X[i,j,k] = vehicle k travels from node i to node j
    x = {}
    for k_idx in range(k):
        for i, j in arcs:
            # No self-loops and no parcel drop to pickup arcs
            if i not in (j, j + m):
                x[i, j, k_idx] = model.createVar(vtype="B", name=f"X_{i}_{j}_{k_idx}")

    # Continuous variables: tau[k,i] = timestamp of vehicle k at node i
    tau = {}
    for k_idx in range(k):
        for i in range(num_nodes):
            tau[k_idx, i] = model.createVar(
                lb=0.0, ub=max_tau, vtype="C", name=f"tau_{k_idx}_{i}"
            )
        model.addConstr(tau[k_idx, 0] == 0.0, name=f"tau_start_depot_{k_idx}")

    # Continuous variables: w[k,i] = parcel load on vehicle k after visiting node i
    w = {}
    for k_idx in range(k):
        for i in range(num_nodes):
            w[k_idx, i] = model.createVar(
                lb=0.0, ub=w_ki[k_idx, i], vtype="C", name=f"w_{k_idx}_{i}"
            )
        model.addConstr(w[k_idx, 0] == 0.0, name=f"w_start_depot_{k_idx}")

    # Continuous variable: z = maximum route cost (objective)
    z = model.createVar(lb=0.0, vtype="C", name="z")

    # ============================================================================
    # Phase 4: Objective Function
    # ============================================================================

    # Minimize the maximum route cost across all vehicles
    model.setObjective(z, sense="min")

    # ============================================================================
    # Phase 5: Constraints
    # ============================================================================

    # Formulation Constraint (1): Coverage
    # each passenger pickup and parcel pickup served exactly once
    for i in preproc["V_p_indices"] + preproc["V_l_indices"]:
        model.addConstr(
            model.quicksum(x[i, j, ik] for j in nodes_from[i] for ik in range(k)) == 1,
            name=f"cov_{i}",
        )

    # Formulation Constraint (2): Parcel pairing - pickup j and drop j+M paired for each vehicle
    for j_idx, j in enumerate(preproc["V_l_indices"]):
        j_drop = preproc["V_lp_indices"][j_idx]
        for k_idx in range(k):
            model.addConstr(
                model.quicksum(x[i, j, k_idx] for i in nodes_to[j])
                == model.quicksum(x[i, j_drop, k_idx] for i in nodes_to[j_drop]),
                name=f"pair_{j}_{j_drop}_{k_idx}",
            )

    # Formulation Constraint (3): Vehicle start and end
    # each vehicle departs start depot and returns to end depot exactly once
    end_depot = num_nodes - 1
    for k_idx in range(k):
        model.addConstr(
            model.quicksum(x[0, i, k_idx] for i in nodes_from[0]) == 1,
            name=f"start_{k_idx}",
        )
        model.addConstr(
            model.quicksum(x[i, end_depot, k_idx] for i in nodes_to[end_depot]) == 1,
            name=f"end_{k_idx}",
        )

    # Formulation Constraint (4): No entry to start depot and no exit from end depot
    # This constraint is optimized out since arcs do not include these transitions
    # for k_idx in range(k):
    #     model.addConstr(
    #         model.quicksum(x[i, 0, k_idx] for i in nodes_to[0]) == 0,
    #         name=f"no_entry_{k_idx}",
    #     )
    #     model.addConstr(
    #         model.quicksum(x[end_depot, i, k_idx] for i in nodes_from[end_depot]) == 0,
    #         name=f"no_exit_{k_idx}",
    #     )

    # Formulation Constraint (5): Flow conservation - for each non-depot node
    for i in range(1, num_nodes - 1):
        for k_idx in range(k):
            model.addConstr(
                model.quicksum(x[i, j, k_idx] for j in nodes_from[i])
                == model.quicksum(x[j, i, k_idx] for j in nodes_to[i]),
                name=f"flow_{i}_{k_idx}",
            )

    # Objective definition: z >= cost of each vehicle's route
    for k_idx in range(k):
        model.addConstr(
            z >= model.quicksum(d_transformed[i, j] * x[i, j, k_idx] for i, j in arcs),
            name=f"maxcost_{k_idx}",
        )

    # Formulation Constraint (8): Weight bounds per vehicle per node
    for k_idx in range(k):
        for i in range(num_nodes):
            lower_bound = max(0.0, preproc["q"][i])
            upper_bound = min(
                float(problem.Q[k_idx]), float(problem.Q[k_idx]) + preproc["q"][i]
            )
            model.addConstr(
                w[k_idx, i] >= lower_bound,
                name=f"w_lower_{k_idx}_{i}",
            )
            model.addConstr(
                w[k_idx, i] <= upper_bound,
                name=f"w_upper_{k_idx}_{i}",
            )

    # Formulation Constraint (9): Parcel precedence - pickup must come before drop
    for j_idx, j in enumerate(preproc["V_l_indices"]):
        j_drop = preproc["V_lp_indices"][j_idx]
        for k_idx in range(k):
            model.addConstr(
                tau[k_idx, j] <= tau[k_idx, j_drop],
                name=f"parcel_prec_{k_idx}_{j}_{j_drop}",
            )

    # Formulation Constraint (10): Linearized timestamp ordering ensures increasing order
    # First, we used linearization with M_tau and m_tau, but now we use indicator constraints
    m_tau = preproc["m_tau"]
    # m_1 = preproc["M_1"]
    # for k_idx in range(k):
    #     for i, j in arcs:
    #         model.addConstr(
    #             tau[k_idx, j] + m_1 * (1 - x[i, j, k_idx]) >= m_tau + tau[k_idx, i],
    #             name=f"tau_order_{k_idx}_{i}_{j}",
    #         )
    for k_idx in range(k):
        for i, j in arcs:
            model.addConstr(
                (x[i, j, k_idx] == 1) >> (tau[k_idx, j] >= tau[k_idx, i] + m_tau)
            )

    # Formulation Constraint (11): Linearized weight validity ensures weight accumulation
    # Similarly, we use indicator constraints now
    # for k_idx in range(k):
    #     for i, j in arcs:
    #         model.addConstr(
    #             w[k_idx, j] - w[k_idx, i]
    #             >= preproc["q"][i] + preproc["W_ki"][k_idx, i] * (x[i, j, k_idx] - 1),
    #             name=f"w_flow_{k_idx}_{i}_{j}",
    #         )
    for k_idx in range(k):
        for i, j in arcs:
            model.addConstr(
                (x[i, j, k_idx] == 1) >> (w[k_idx, j] >= w[k_idx, i] + preproc["q"][i])
            )
    model.update()
    # ============================================================================
    # Phase 6: Pre-Optimization Model Size Check
    # ============================================================================

    print("\n=== PRE-OPTIMIZATION MODEL SIZE ===", flush=True)
    print(f"Total variables: {model.NumVars}", flush=True)
    print(f"Total constraints: {model.NumConstrs}", flush=True)

    model.optimize()
    print("Finishing optimizing")
    if verbose:
        # Debug output to stdout with flush
        print("=== SOLVER MODEL ===", flush=True)
        print(f"Total variables: {model.NumVars}", flush=True)
        print(f"Total constraints: {model.NumConstrs}", flush=True)

        print("\n=== SOLUTION VALUES ===", flush=True)
        print(f"Status: {model.status}", flush=True)
        print(
            f"Objective: {model.objVal if model.solCount > 0 else 'No solution'}\n",
            flush=True,
        )

        if model.solCount > 0:
            print("=== X VARIABLES (Arc usage) ===", flush=True)
            for v in model.getVars():
                if v.VarName.startswith("X_") and v.X > 0.5:
                    print(f"{v.VarName} = {v.X}", flush=True)

            print("\n=== TAU VARIABLES (Timestamps) ===", flush=True)
            for v in model.getVars():
                if v.VarName.startswith("tau_") and v.X > 0:
                    print(f"{v.VarName} = {v.X}", flush=True)

            print("\n=== W VARIABLES (Parcel loads) ===", flush=True)
            for v in model.getVars():
                if v.VarName.startswith("w_") and v.X > 0:
                    print(f"{v.VarName} = {v.X}", flush=True)

    elapsed_time = time.time() - start_time

    # ============================================================================
    # Phase 7: Extract Results
    # ============================================================================

    # Populate info dictionary with solver statistics
    info_dict["elapsed_time"] = elapsed_time
    info_dict["solver"] = f"{solver.capitalize()} MILP"
    info_dict["time_limit"] = time_limit

    # Map solver-agnostic status to readable format
    optimal_code = model.get_status_code("OPTIMAL")
    time_limit_code = model.get_status_code("TIME_LIMIT")
    infeasible_code = model.get_status_code("INFEASIBLE")

    if model.status == optimal_code:
        info_dict["status"] = "optimal"
    elif model.status == time_limit_code:
        info_dict["status"] = "time_limit"
    elif model.status == infeasible_code:
        info_dict["status"] = "infeasible"
    else:
        info_dict["status"] = f"status_{model.status}"

    if model.solCount > 0:
        info_dict["objective_value"] = model.objVal
        info_dict["gap"] = model.MIPGap if model.status != optimal_code else 0.0
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


if __name__ == "__main__":
    # TODO: Retrieve Gurobi Academic License before running tests
    # solver = AlgoSolver(milp)
    # sols2, gaps2, msg2 = attempt_dataset(
    #     solver, "H", note="test MILP on H dataset", verbose=True
    # )
    # summarize_dataset("H", verbose=True)
    problems = exact_problems
    passed_tests = 0
    for prob in problems:
        algo_solver = AlgoSolver(milp, {"time_limit": 600.0, "verbose": False})
        # try:
        sol, info = algo_solver.solve(prob)
        if sol is None or not sol.is_valid(True):
            print(f"Problem {prob.name} - No valid solution found. Info: {info}")
        else:
            passed_tests += 1
        # except Exception as exc:
        #     print(f"Problem {prob.name} - Exception during solving", exc)
    print(f"Passed {passed_tests} out of {len(problems)} tests.")
