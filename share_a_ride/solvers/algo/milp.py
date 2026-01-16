"""
MILP solver for Share-a-Ride Problem using Gurobi.
Implements Groups 1 & 2: Route definition, coverage, and flow continuity.
"""

import time
from typing import Any, Optional, Tuple, Dict

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:
    raise ImportError(
        "Gurobi is required for MILP solver. Install it with: pip install gurobipy"
    ) from exc

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution


def milp_solver(
    problem: ShareARideProblem,
    verbose: bool = False,
    time_limit: int = 300,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    MILP solver for SARP using Gurobi.
    Minimizes the maximum route cost across all vehicles.

    Params:
        - problem: ShareARideProblem instance
        - verbose: Print Gurobi output if True
        - time_limit: Time limit in seconds for solver (default 300)

    Returns:
        (sol, info): tuple where
        - sol: Solution object if optimal/feasible solution found, else None
        - info: Dictionary with:
            + status: "optimal", "feasible", "infeasible", "timeout"
            + time: Wall clock time
            + objval: Objective value (max route cost) if feasible
            + gap: Optimality gap if not optimal
            + num_routes_used: Number of vehicles actually used
    """
    start_time = time.time()

    try:
        # Create Gurobi model
        model = gp.Model("SARP_MILP")
        if not verbose:
            model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit

        # Create variables and add constraints
        x = _create_arc_variables(model, problem)
        z = model.addVar(vtype=GRB.CONTINUOUS, name="max_cost")

        # Add constraint groups
        _add_coverage_constraints(model, x, problem)
        _add_depot_balance_constraints(model, x, problem)
        _add_flow_conservation_constraints(model, x, problem)

        # Add objective: minimize max route cost
        _add_objective(model, x, z, problem)
        model.update()

        # Optimize
        model.optimize()

        # Extract solution
        elapsed = time.time() - start_time
        sol, info = _extract_solution(model, x, problem, elapsed)

        return sol, info

    except gp.GurobiError as e:
        elapsed = time.time() - start_time
        return None, {"status": "error", "time": elapsed, "error": str(e)}


def _create_arc_variables(model: gp.Model, problem: ShareARideProblem) -> Dict:
    """
    Create binary arc variables x[i,j,k] for all edges and vehicles.

    Returns:
        x: Dictionary x[(i,j,k)] = Gurobi variable, binary
    """
    x = {}
    V = problem.num_nodes
    K = problem.K

    for k in range(K):
        for i in range(V):
            for j in range(V):
                if i != j:
                    var_name = f"x_{i}_{j}_{k}"
                    x[(i, j, k)] = model.addVar(vtype=GRB.BINARY, name=var_name)

    return x


def _add_coverage_constraints(
    model: gp.Model, x: Dict, problem: ShareARideProblem
) -> None:
    """
    GROUP 1: Coverage constraints

    C1a: Each non-depot node has exactly one incoming arc
    C1b: Each non-depot node has exactly one outgoing arc
    """
    V = problem.num_nodes
    K = problem.K

    # C1a: In-degree = 1 for each non-depot node
    for j in range(1, V):  # exclude depot (node 0)
        in_arcs = gp.quicksum(
            x[(i, j, k)] for i in range(V) for k in range(K) if i != j
        )
        model.addConstr(in_arcs == 1, name=f"in_degree_{j}")

    # C1b: Out-degree = 1 for each non-depot node
    for i in range(1, V):  # exclude depot (node 0)
        out_arcs = gp.quicksum(
            x[(i, j, k)] for j in range(V) for k in range(K) if i != j
        )
        model.addConstr(out_arcs == 1, name=f"out_degree_{i}")


def _add_depot_balance_constraints(
    model: gp.Model, x: Dict, problem: ShareARideProblem
) -> None:
    """
    GROUP 2: Depot balance constraints

    C2a: Each vehicle departs depot at most once
    C2b: Each vehicle returns to depot at most once
    C2c: Vehicles that depart must return (in-degree = out-degree at depot)
    """
    V = problem.num_nodes
    K = problem.K
    depot = 0

    # C2a: Each vehicle leaves depot at most once
    for k in range(K):
        depart = gp.quicksum(x[(depot, j, k)] for j in range(1, V))
        model.addConstr(depart <= 1, name=f"depart_{k}")

    # C2b: Each vehicle returns to depot at most once
    for k in range(K):
        arrive = gp.quicksum(x[(i, depot, k)] for i in range(1, V))
        model.addConstr(arrive <= 1, name=f"arrive_{k}")

    # C2c: Depot balance: if depart, must return
    for k in range(K):
        depart = gp.quicksum(x[(depot, j, k)] for j in range(1, V))
        arrive = gp.quicksum(x[(i, depot, k)] for i in range(1, V))
        model.addConstr(depart == arrive, name=f"depot_balance_{k}")


def _add_flow_conservation_constraints(
    model: gp.Model, x: Dict, problem: ShareARideProblem
) -> None:
    """
    GROUP 2 (cont.): Flow conservation constraints

    C3: Flow conservation at each non-depot node (in-flow = out-flow per vehicle)
    Ensures routes are connected and don't fragment.
    """
    V = problem.num_nodes
    K = problem.K

    for k in range(K):
        for j in range(1, V):  # exclude depot (node 0)
            in_flow = gp.quicksum(x[(i, j, k)] for i in range(V) if i != j)
            out_flow = gp.quicksum(x[(j, l, k)] for l in range(V) if j != l)
            model.addConstr(in_flow == out_flow, name=f"flow_balance_{j}_{k}")


def _add_objective(
    model: gp.Model, x: Dict, z: gp.Var, problem: ShareARideProblem
) -> None:
    """
    Add objective function: minimize max route cost.

    z >= sum_i sum_j D[i][j] * x[i][j][k] for all k
    minimize z
    """
    K = problem.K
    D = problem.D

    for k in range(K):
        # Cost of route k
        route_cost = gp.quicksum(
            D[i][j] * x[(i, j, k)]
            for i in range(problem.num_nodes)
            for j in range(problem.num_nodes)
            if i != j
        )
        # z >= cost_k
        model.addConstr(z >= route_cost, name=f"max_cost_k_{k}")

    # Minimize z
    model.setObjective(z, GRB.MINIMIZE)


def _extract_solution(
    model: gp.Model, x: Dict, problem: ShareARideProblem, elapsed: float
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Extract routes from optimal/feasible solution.

    Returns:
        (sol, info): Solution object (or None) and info dictionary
    """
    info = {
        "time": elapsed,
        "status": _status_to_string(model.Status),
    }

    # Check if solution exists
    if model.SolCount == 0:
        info["objval"] = None
        info["gap"] = None
        info["num_routes_used"] = 0
        return None, info

    # Extract solution values
    info["objval"] = model.ObjVal
    if model.Status != GRB.OPTIMAL:
        info["gap"] = model.MIPGap
    else:
        info["gap"] = 0.0

    # Build routes from x values
    routes = _build_routes(x, problem)
    num_routes_used = sum(
        1 for r in routes if len(r) > 2
    )  # routes with nodes besides depot
    info["num_routes_used"] = num_routes_used

    # Create Solution object
    try:
        sol = Solution(problem, routes)
        if not sol.is_valid():
            info["status"] = "infeasible_solution"
            return None, info
        return sol, info
    except Exception as e:
        info["status"] = "extraction_error"
        info["error"] = str(e)
        return None, info


def _build_routes(x: Dict, problem: ShareARideProblem) -> list:
    """
    Reconstruct routes from arc variables.

    For each vehicle, traverse the path starting from depot.
    """
    routes = [[] for _ in range(problem.K)]
    visited = [[False] * problem.num_nodes for _ in range(problem.K)]

    for k in range(problem.K):
        # Start from depot
        current = 0
        route = [0]
        visited[k][0] = True

        while True:
            # Find next node from current
            next_node = None
            for j in range(problem.num_nodes):
                if j != current and not visited[k][j]:
                    if (current, j, k) in x and x[(current, j, k)].X > 0.5:
                        next_node = j
                        break

            if next_node is None:
                # Try to return to depot
                if (current, 0, k) in x and x[(current, 0, k)].X > 0.5:
                    route.append(0)
                break
            else:
                route.append(next_node)
                visited[k][next_node] = True
                current = next_node

        routes[k] = route

    return routes


def _status_to_string(gurobi_status: int) -> str:
    """Convert Gurobi status code to string."""
    status_map = {
        GRB.OPTIMAL: "optimal",
        GRB.SUBOPTIMAL: "feasible",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "infeasible_or_unbounded",
        GRB.UNBOUNDED: "unbounded",
    }
    return status_map.get(gurobi_status, "unknown")
