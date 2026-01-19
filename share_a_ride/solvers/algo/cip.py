"""
Constraint Integer Programming (CIP) solver for the Share-a-Ride Problem (SARP).

This module implements an exact CP-SAT formulation (Google OR-Tools) for the SARP
objective: minimize the maximum route length among the K taxis.

Key SARP constraints (see description.txt):
- Every passenger pickup i must be followed immediately by its drop-off i+N+M.
- Parcel pickup/delivery pairing with precedence.
- Parcel capacity constraints per vehicle.
"""
import time
from typing import Any, Dict, Optional, Tuple

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "OR-Tools is not installed. Install it separately: pip install ortools"
    ) from exc

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.solvers.algo.bnb import mst_lowerbound




def _compute_mst_bound(problem: ShareARideProblem) -> int:
    """
    Compute the MST lower bound on total cost to serve all requests.

    We build action nodes for all requests (passengers and parcels) plus depot,
    and compute the MST lower bound. This is a lower bound on the total travel
    cost across all vehicles.
    """
    dmat = problem.D

    # Build action nodes: (entry_node, exit_node) for each request.
    # Depot: (0, 0)
    # Passenger pid: pickup -> drop is atomic, so (pickup, drop)
    # Parcel lid: pickup and drop are separate actions, (pick, pick) and (drop, drop)
    action_nodes = [(0, 0)]  # depot

    for pid in range(1, problem.N + 1):
        pick, drop = problem.pserve(pid)
        action_nodes.append((pick, drop))

    for lid in range(1, problem.M + 1):
        pick = problem.lpick(lid)
        drop = problem.ldrop(lid)
        action_nodes.append((pick, pick))
        action_nodes.append((drop, drop))

    return mst_lowerbound(action_nodes, dmat)




def _build_x_variables(
    model: cp_model.CpModel, problem: ShareARideProblem
) -> Dict[tuple[int, int, int], cp_model.IntVar]:
    """
    Create arc variables x[i,j,k] with SARP-specific pruning.
    We follow 3-index notation: x[i,j,k] == 1 if vehicle k travels from node i to node j.
    1. Depot self-loops x[0,0,k] allowed to represent unused vehicles.
    2. Passenger direct trips: pickup -> drop only; drop can only be entered from its pickup.
    3. Capacity pruning: vehicle k cannot pick up parcels heavier than its capacity.
    """
    n_nodes = problem.num_nodes
    x: Dict[tuple[int, int, int], cp_model.IntVar] = {}

    for k in range(problem.K):
        cap_k = int(problem.Q[k])
        for i in range(n_nodes):
            for j in range(n_nodes):
                # Allow depot self-loop to represent an unused vehicle.
                if i == j:
                    if i == 0:
                        x[(0, 0, k)] = model.NewBoolVar(f"x_0_0_{k}")
                    continue

                # Passenger direct trip: pickup -> corresponding drop only.
                if problem.is_ppick(i):
                    pid = problem.rev_ppick(i)
                    _pick, drop = problem.pserve(pid)
                    if j != drop:
                        continue

                # Passenger direct trip: drop can only be entered from its pickup.
                if problem.is_pdrop(j):
                    pid = problem.rev_pdrop(j)
                    pick, _drop = problem.pserve(pid)
                    if i != pick:
                        continue

                # Capacity pruning: vehicle k can never pick up a too-heavy parcel.
                if problem.is_lpick(j):
                    lid = problem.rev_lpick(j)
                    if int(problem.q[lid - 1]) > cap_k:
                        continue

                x[(i, j, k)] = model.NewBoolVar(f"x_{i}_{j}_{k}")


    return x




def _extract_solution(
    problem: ShareARideProblem,
    solver: cp_model.CpSolver,
    x_dict: Dict[tuple[int, int, int], cp_model.IntVar],
    num_nodes: int,
    k: int,
) -> Solution:
    """
    Extract routes from the ortools solver to the standard Solution encapsulation.
    """
    routes: list[list[int]] = []
    for v in range(k):
        succ: Dict[int, int] = {}
        for (i, j, vk), var in x_dict.items():
            if vk != v:
                continue
            if solver.BooleanValue(var):
                succ[i] = j

        route = [0]
        current = 0
        guard = 0
        while True:
            nxt = succ.get(current, 0)
            route.append(nxt)
            current = nxt
            guard += 1
            if current == 0:
                break
            if guard > num_nodes + 5:
                # Should not happen in a feasible solution; prevents infinite loops.
                break
        routes.append(route)

    return Solution(problem=problem, routes=routes)




def cip_solver(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution] = None,      # pylint: disable=unused-argument
    incumbent: Optional[int] = None,
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
    **_kwargs: Any,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Solve SARP via CP-SAT (CIP model).

    Args:
        problem: ShareARideProblem instance to solve.
        partial: Optional partial solution to build upon (not supported here).
        incumbent: Optional incumbent objective value to prune search.
        seed: Optional random seed for reproducibility.
        time_limit: Time limit in seconds.
        verbose: Whether to log solver progress.
    
    Returns:
        A tuple (solution, info_dict) where solution is a Solution instance or None if
        no solution was found, and info_dict contains solver statistics.
    """
    start = time.time()

    if incumbent is None:
        incumbent = 10**18      # Effectively unlimited.


    # ---------------- Model ----------------
    model = cp_model.CpModel()
    n_nodes = problem.num_nodes
    m = problem.M
    k = problem.K
    dmat = problem.D


    # ---------------- Variables ----------------
    # 3-index arc variables.
    x_dict = _build_x_variables(model, problem)
    # served[v,node]: whether vehicle v serves node
    served: Dict[tuple[int, int], cp_model.IntVar] = {}
    # pos[v,node]: position of node in vehicle v's route.
    pos: Dict[tuple[int, int], cp_model.IntVar] = {}
    # load[v,node]: load of vehicle v after visiting node.
    load: Dict[tuple[int, int], cp_model.IntVar] = {}

    for ve in range(k):
        cap_v = int(problem.Q[ve])
        for node in range(1, n_nodes):
            served[(ve, node)] = model.NewBoolVar(f"served_{ve}_{node}")
            pos[(ve, node)] = model.NewIntVar(0, n_nodes, f"pos_{ve}_{node}")
            load[(ve, node)] = model.NewIntVar(0, cap_v, f"load_{ve}_{node}")

        # Depot load fixed to 0.
        load[(ve, 0)] = model.NewIntVar(0, 0, f"load_{ve}_0")

    # Compute bounds on route distance using MST heuristic.
    # MST gives a lower bound on total cost to serve all requests.
    mst_lower = _compute_mst_bound(problem) // k
    mst_upper = 2 * mst_lower

    dist_upper = min(incumbent, mst_upper)
    dist_lower = mst_lower

    dist: Dict[int, cp_model.IntVar] = {}
    for ve in range(k):
        dist[ve] = model.NewIntVar(0, dist_upper, f"dist_{ve}")
    z = model.NewIntVar(dist_lower, dist_upper, "Z")


    # ---------------- Constraints ----------------
    # (1) Coverage: every non-depot node has exactly one incoming arc across all vehicles.
    for j in range(1, n_nodes):
        incoming = [
            x_dict[(i, j, v)]
            for v in range(k)
            for i in range(n_nodes)
            if (i, j, v) in x_dict
        ]
        model.Add(sum(incoming) == 1)

    # (2) Depot: each vehicle either does one tour or stays at depot (self-loop).
    for ve in range(k):
        out_of_depot = [x_dict[(0, j, ve)] for j in range(n_nodes) if (0, j, ve) in x_dict]
        into_depot = [x_dict[(i, 0, ve)] for i in range(n_nodes) if (i, 0, ve) in x_dict]
        model.Add(sum(out_of_depot) == 1)
        model.Add(sum(into_depot) == 1)

    # (3) Flow conservation per vehicle on non-depot nodes.
    for ve in range(k):
        for node in range(1, n_nodes):
            in_vars = [x_dict[(i, node, ve)] for i in range(n_nodes) if (i, node, ve) in x_dict]
            out_vars = [x_dict[(node, j, ve)] for j in range(n_nodes) if (node, j, ve) in x_dict]
            model.Add(sum(in_vars) == sum(out_vars))
            model.Add(sum(in_vars) == served[(ve, node)])

            # Position is 0 if not served, in [1..] if served.
            model.Add(pos[(ve, node)] == 0).OnlyEnforceIf(served[(ve, node)].Not())
            model.Add(pos[(ve, node)] >= 1).OnlyEnforceIf(served[(ve, node)])

    # (4) Parcel pairing: pickup and drop must be on the same vehicle.
    for lid in range(1, m + 1):
        pick = problem.lpick(lid)
        drop = problem.ldrop(lid)
        for ve in range(k):
            model.Add(served[(ve, pick)] == served[(ve, drop)])

    # (5) Subtour elimination (MTZ) using position variables.
    big_m = n_nodes + 1
    for ve in range(k):
        for i in range(1, n_nodes):
            for j in range(1, n_nodes):
                if i == j:
                    continue
                if (i, j, ve) not in x_dict:
                    continue
                model.Add(pos[(ve, i)] + 1 <= pos[(ve, j)] + big_m * (1 - x_dict[(i, j, ve)]))

    # (6) Parcel precedence: pickup must occur before drop if served by the same vehicle.
    for lid in range(1, m + 1):
        pick = problem.lpick(lid)
        drop = problem.ldrop(lid)
        for ve in range(k):
            model.Add(pos[(ve, pick)] + 1 <= pos[(ve, drop)]).OnlyEnforceIf(served[(ve, pick)])

    # (7) Capacity propagation along arcs (load after visiting each node).
    for ve in range(k):
        cap_v = int(problem.Q[ve])
        for i in range(n_nodes):
            for j in range(n_nodes):
                if (i, j, ve) not in x_dict:
                    continue

                # Load delta at node j (parcels only).
                if problem.is_lpick(j):
                    lid = problem.rev_lpick(j)
                    delta_j = int(problem.q[lid - 1])
                elif problem.is_ldrop(j):
                    lid = problem.rev_ldrop(j)
                    delta_j = -int(problem.q[lid - 1])
                else:
                    delta_j = 0

                li = load[(ve, i)] if (ve, i) in load else load[(ve, 0)]
                lj = load[(ve, j)] if (ve, j) in load else load[(ve, 0)]

                # lj == li + delta_j when arc is used (linearized with big-M).
                model.Add(lj >= li + delta_j - cap_v * (1 - x_dict[(i, j, ve)]))
                model.Add(lj <= li + delta_j + cap_v * (1 - x_dict[(i, j, ve)]))

    # (8) Distance per vehicle.
    for ve in range(k):
        terms = []
        for (i, j, ve_k), var in x_dict.items():
            if ve_k != ve:
                continue
            if i == 0 and j == 0:
                continue
            terms.append(int(dmat[i][j]) * var)
        model.Add(dist[ve] == sum(terms))

    # (9) Minimax objective.
    model.AddMaxEquality(z, [dist[v] for v in range(k)])
    if incumbent is not None:
        model.Add(z < incumbent)
    model.Minimize(z)


    # ---------------- Solve ----------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    if seed is not None:
        solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = verbose
    solver.parameters.num_search_workers = 8

    # Solve the model.
    status = solver.Solve(model)
    elapsed = time.time() - start

    info: Dict[str, Any] = {
        "time": elapsed,
        "time_limit": time_limit,
        "solver": "OR-Tools CP-SAT",
        "objective_value": None,
        "status": "unknown",
    }

    if status == cp_model.OPTIMAL:
        info["status"] = "optimal"
        info["objective_value"] = int(solver.ObjectiveValue())
    elif status == cp_model.FEASIBLE:
        # Typically means time limit hit but a solution exists.
        info["status"] = "feasible"
        info["objective_value"] = int(solver.ObjectiveValue())
    elif status == cp_model.INFEASIBLE:
        info["status"] = "infeasible"
    elif status == cp_model.MODEL_INVALID:
        info["status"] = "model_invalid"
    else:
        info["status"] = "unknown"

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = _extract_solution(problem, solver, x_dict, n_nodes, k)
        if not sol.is_valid(False):
            # Should not happen; helps catch modeling/extraction issues.
            return None, {**info, "status": "invalid_solution"}
        return sol, info

    return None, info




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import exact_problems
    from share_a_ride.solvers.algo.greedy import iterative_greedy_solver as init_solver
    from share_a_ride.solvers.algo.bnb import bnb_solver as ref_solver

    for probid, prob in enumerate(exact_problems[:4], start=1):
        init, _info = init_solver(prob, time_limit=10.0, verbose=False, seed=42)
        assert init is not None
        init_cost = init.max_cost
        print(f"Problem {probid}: initial cost = {init_cost}")

        ref, _info = ref_solver(
            prob, incumbent=init_cost + 1, time_limit=100.0, verbose=False, seed=42
        )
        assert ref is not None
        ref_cost = ref.max_cost
        print(f"Problem {probid}: reference cost = {ref_cost}")

        soln, info_dict = cip_solver(
            prob, incumbent=ref_cost + 1, time_limit=30.0, verbose=True, seed=42
        )
        if soln is None:
            print(f"Problem {probid}: no solution ({info_dict.get('status')})")
            continue
        if not soln.is_valid(False):
            print(f"Problem {probid}: invalid solution")
            continue

        if soln.max_cost > ref_cost:
            print(f"Problem {probid}: suboptimal, cost={soln.max_cost}, ref={ref_cost}")
            continue
        print(f"Problem {probid}: ok, cost={soln.max_cost}")
