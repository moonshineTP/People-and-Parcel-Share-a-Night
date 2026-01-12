"""
Weighted A* search solver for Share-a-Ride.
"""
import heapq
import time

from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Dict, List, Optional, Tuple, ParamSpec

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, PartialSolutionSwarm, Solution
from share_a_ride.solvers.algo.greedy import iterative_greedy_solver
from share_a_ride.solvers.algo.utils import (
    Action, balanced_scorer, apply_general_action, enumerate_actions_greedily
)
from share_a_ride.solvers.operator.relocate import relocate_operator


# ///////// Type Alias ////////
Params = ParamSpec('Params')
CostFunction = Callable[Concatenate[PartialSolution, Params], float]
PredFunction = Callable[Concatenate[PartialSolution, Params], float]
WeightFunction = Callable[Concatenate[PartialSolution, Params], float]
DefensePolicy = Callable[Concatenate[PartialSolution, Params], Optional[Solution]]




# //////// Function Components ////////
def _default_cost_function(partial: PartialSolution, seed: int) -> float:
    """
    Default cost ``g``: we choose the current maximum route cost.
    """
    return balanced_scorer(partial, seed=seed)


def _default_pred_function(partial: PartialSolution) -> float:
    """
    Admissible lower bound on remaining work from ``partial``.

    This heuristic exploits several SARP / VRP-with-pickup-and-delivery
    properties while remaining *optimistic* (for A* admissibility):

    - Each route currently has a head position; any new work for that
      route must start from there (never cheaper than from the depot).
    - All unserved requests (pickups and drops) must eventually be visited.
    - These unserved locations together induce a metric TSP-like structure,
      for which a minimum spanning tree (MST) gives a natural routing
      lower bound.

    To keep the heuristic cheap yet tighter than a plain MST-on-nodes,
    we do the following conservative construction:

    1. We identify "extendable nodes" which include all action nodes
       plus active route heads that are currently at a pickup location.
    2. We run a Prim-style MST over these nodes using symmetric edge
       weights derived from the matrix ``D``.
    3. We add lower bounds for connecting the depot to the start of new
       routes and for returning from the end of routes to the depot.

    Because all ingredients are built from pointwise minima of actual
    path costs in ``D`` and ignore capacity conflicts (which can only
    make the true completion more expensive), the resulting value is
    admissible.
    """
    problem = partial.problem
    D = problem.D   # pylint: disable=invalid-name
    K = problem.K               # pylint: disable=invalid-name


    # //// Identify extendable nodes for MST
    # Count active/inactive routes
    non_started_cnt = K
    non_returned_cnt = K
    active_heads: List[Tuple[int, int]] = []
    extendable_nodes: List[Tuple[int, int]] = []    # List of action node (physical head-tail)
    internal_service_cost = 0

    # Extract extendable_nodes
    for pid in partial.remaining_pass_serve:
        phys_fr, phys_to = problem.pserve(pid)
        internal_service_cost += D[phys_fr][phys_to]
        extendable_nodes.append((phys_fr, phys_to))
    for lid in partial.remaining_parc_pick:
        phys_fr, phys_to = problem.lpick(lid), problem.lpick(lid)
        extendable_nodes.append((phys_fr, phys_to))
    for lid in partial.remaining_parc_drop:
        phys_fr, phys_to = problem.ldrop(lid), problem.ldrop(lid)
        extendable_nodes.append((phys_fr, phys_to))

    # Extract heads and route state count
    for state in partial.states:
        if state["actions"] > 0:    # more than depot only -> active
            non_started_cnt -= 1
            head_node = state["pos"]

            if problem.is_pdrop(head_node):
                active_heads.append((head_node - 1, head_node))
            else:
                active_heads.append((head_node, head_node))

        if state["ended"]:
            non_returned_cnt -= 1

    all_nodes = extendable_nodes + active_heads


    # //// Compute extendable nodes lower bound
    # We use a MST lower bund on the extendable nodes with
    # symmetric edge costs defined as min(D[i][j], D[j][i]).
    if len(extendable_nodes) <= 1:
        mst_weight = 0
    else:
        # Helper for asymmetric physical node due to atomic passenger service
        def get_dist(u, v):
            return min(D[u[1]][v[0]], D[v[1]][u[0]])


        # //// Prim's algorithm
        # Initialize variables
        current = 0                 # the frontier node, initialized to 0
        remaining = set(range(1, len(all_nodes)))# exclude first node
        best: Dict[int, int] = {}   # best cost to frontier
        # Initialize best costs
        for act in remaining:
            actu, actv = all_nodes[current], all_nodes[act]
            best[act] = min(get_dist(actu, actv), get_dist(actv, actu))

        # Build MST
        mst_weight = 0
        while remaining:
            # Select the next node with the smallest edge cost to the MST
            next_idx = min(remaining, key=lambda act: best[act])
            edge_cost = best[next_idx]
            mst_weight += edge_cost
            remaining.remove(next_idx)

            # Update frontier
            actu = all_nodes[next_idx]
            for act in remaining:
                actv = all_nodes[act]
                cost_uv = min(get_dist(actu, actv), get_dist(actv, actu))
                if cost_uv < best[act]:
                    best[act] = cost_uv


    # //// Compute start/end depot lower bound
    # Opening edges: from depot 0 to each possible start node.
    start_edge_costs = [D[0][phys_fr] for phys_fr, _ in extendable_nodes]
    start_lb = sum(sorted(start_edge_costs)[:non_started_cnt])

    # Closing edges: from each possible return node back to depot 0.
    return_edge_costs = [D[phys_to][0] for _, phys_to in all_nodes]
    return_lb = sum(sorted(return_edge_costs)[:non_returned_cnt])


    # //// Finalize
    lb = mst_weight + start_lb + return_lb + internal_service_cost

    return float(lb) // problem.K


def _default_weight_function(partial: PartialSolution) -> float:
    """
    Default dynamic weight ``w`` based on depth.
    This is a simple adaptive weight that encourages deeper nodes.
    """
    terminal = partial.problem.num_actions
    depth = partial.num_actions
    return 1 - float(depth) / float(terminal) if terminal > 0 else 0.0


def _default_defense_policy(
        partial: PartialSolution,
        time_limit: float = 20.0,
        seed: Optional[int] = None,
        verbose: bool = False
    ) -> Optional[Solution]:
    """
    Defense policy: complete a promising partial with beam search.
    This mirrors the default defense strategy used in ``mcts.py``.
    """

    best, _ = iterative_greedy_solver(
        partial.problem,
        partial,
        iterations=2000,
        time_limit=20.0,
        seed=113*seed if seed is not None else None,
        verbose=verbose,
    )
    return best




# ============= A* core implementation =============
@dataclass
class AStarNode:
    """Internal node representation for weighted A* search.
    Each node stores the partial solution and its scalar scores:
    * ``g`` - accumulated cost from the root according to ``cost_function``.
    * ``h`` - heuristic prediction of the remaining cost.
    * ``w`` - dynamic weight factor used in the combined objective.
    * ``f`` - weighted A* priority ``f = g + (1 + eps * w) * h``.
    """

    partial: PartialSolution
    g: float
    h: float
    w: float
    f: float

    def __lt__(self, other: "AStarNode") -> bool:
        return self.f < other.f


def _astar_priority(
    partial: PartialSolution,
    eps: float,
    cost_function: CostFunction,
    pred_function: PredFunction,
    weight_function: WeightFunction,
    seed: Optional[int],
) -> Tuple[float, float, float, float]:
    """
    Compute ``(f, g, h, w)`` for a partial solution.
    """

    g = float(cost_function(partial, seed))
    h = float(pred_function(partial))
    w = float(weight_function(partial))
    f = g + (1.0 + eps * w) * h
    return f, g, h, w




def astar_enumerator(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        n_return: int = 10,
        eps: float = 0.285,
        width: int = 5,
        cutoff_depth: int = 5,
        cost_function: CostFunction = _default_cost_function,
        pred_function: PredFunction = _default_pred_function,
        weight_function: WeightFunction = _default_weight_function,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[PartialSolutionSwarm, Dict[str, Any]]:
    """Run weighted A* and return the best ``k`` leaf partial solutions.

    This function executes a resource-constrained weighted A* search with:
    - ``eps`` to balance exploration and exploitation
    - ``width`` to limit branching
    - periodic cutoff mechanism in ``cutoff_depth`` to prune the open set.
    - ``cost_function``, ``pred_function``, ``weight_function`` to
        customize the A* scoring.

    The enumerator collects the best ``k`` partial solutions encountered
    during the search. The collection priority is hierarchical:
    1. Depth (deeper solutions are preferred).
    2. Cost (lower cost solutions are preferred).
    3. Insertion order (earlier solutions are preferred).
    """
    start_time = time.time()

    # Fill in defaults
    if n_return <= 0:
        raise ValueError("k must be positive")
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    # Push root node
    open_heap: List[Tuple[float, AStarNode]] = []
    collected_heap: List[Tuple[int, int, int, PartialSolution]] = []
    collected_count = 0
    f0, g0, h0, w0 = _astar_priority(
        partial,
        eps,
        cost_function,
        pred_function,
        weight_function,
        seed=41*seed if seed is not None else None
    )
    root = AStarNode(partial=partial, g=g0, h=h0, w=w0, f=f0)
    heapq.heappush(open_heap, (root.f, root))


    # //// Helper to push collected leaves
    def _push_collected(ps: PartialSolution) -> None:
        nonlocal collected_count

        # Update counter for tie-breaking
        collected_count += 1

        # Push into collected heap
        depth = ps.num_actions
        cost = ps.max_cost
        heapq.heappush(collected_heap, (-depth, cost, collected_count, ps))

        # Maintain size limit
        if len(collected_heap) > n_return:
            heapq.heappop(collected_heap)


    # //// Helper to expand a node and push into open set
    def _expand_and_push_open(ps: PartialSolution, action: Action):
        child_ps = ps.copy()
        apply_general_action(child_ps, action)
        f, g, h, w = _astar_priority(
            ps,
            eps,
            cost_function,
            pred_function,
            weight_function,
            seed=73*seed if seed else None
        )
        node = AStarNode(partial=ps, g=g, h=h, w=w, f=f)
        heapq.heappush(open_heap, (node.f, node))


    # //// Main loop
    iterations = 0
    status = "done"
    while open_heap:
        if time_limit is not None and time.time() - start_time >= time_limit:
            status = "overtime"
            if verbose:
                print(f"[A*] Time limit reached after {iterations} iterations")
            break

        # Logging
        if verbose:
            elapsed = time.time() - start_time

            # Log at 1, 2, ..., 10, 20, ..., 100, 200, ...
            magnitude = 10 ** (len(str(iterations)) - 1)
            if iterations % magnitude == 0:
                deepest = -collected_heap[0][0]
                deepest_costs: List[int] = [
                    entry[1]
                    for entry in collected_heap
                    if -entry[0] == deepest
                ]
                if not deepest_costs:
                    continue
                min_cost = min(deepest_costs)
                max_cost = max(deepest_costs)
                min_f = open_heap[0][0]
                max_f = open_heap[-1][0]
                print(
                    f"[A*] Iteration {iterations}, "
                    f"heap_size: {len(open_heap) + 1}, "
                    f"depth {deepest}, "
                    f"cost range {min_cost} - {max_cost}, "
                    f"f range {min_f:.2f} - {max_f:.2f}, "
                    f"time={elapsed:.2f}s",
                )

        # Pop best node
        _, node = heapq.heappop(open_heap)
        ps = node.partial
        _push_collected(ps)
        iterations += 1

        # Cutoff: empty the heap at each cutoff depth
        if ps.num_actions % cutoff_depth == 0 and ps.num_actions > 0:
            open_heap = []

        # Check for terminality.
        if ps.is_completed():
            continue

        # Expand children
        actions = enumerate_actions_greedily(ps, None)
        for action in actions[:width]:
            _expand_and_push_open(ps, action)

    # Collect results
    collected_partial = [ps[3] for ps in collected_heap]
    swarm = PartialSolutionSwarm(collected_partial)

    # Summary
    stats: Dict[str, Any] = {
        "iterations": iterations,
        "time": time.time() - start_time,
        "collected": len(collected_heap),
        "status": status,
    }

    # Logging
    if verbose:
        print(
            f"[A*] Completed enumeration - collected {stats['collected']} leaves "
            f"in {stats['time']:.3f}s (iterations={stats['iterations']}).",
        )
        print(f"[A*] Best partial solutions cost: {collected_heap[0][0]}")
        print()

    return swarm, stats




def astar_solver(
        problem: ShareARideProblem,
        partial: Optional[PartialSolution] = None,
        eps: float = 0.285,
        width: int = 5,
        cutoff_depth: int = 5,
        cutoff_size: int = 1000,
        cutoff_ratio: float = 0.286,
        cost_function: CostFunction = _default_cost_function,
        pred_function: PredFunction = _default_pred_function,
        weight_function: WeightFunction = _default_weight_function,
        defense_policy: DefensePolicy = _default_defense_policy,
        time_limit: float = 30.0,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """Weighted A* solver returning the best complete solution.

    This function executes a resource-constrained weighted A* search to find
    a complete solution for the Share-a-Ride problem. It balances exploration
    and exploitation using a weighted A* scheme and manages memory/time
    complexity via beam search and periodic pruning.

    Key parameters and their roles:
    - ``eps``: Controls the weight of the heuristic function in the A* priority
      calculation (f = g + (1 + eps * w) * h). Higher values encourage
      greedier, depth-first exploration.
    - ``width``: Limits the branching factor at each node. Only the top
      ``width`` actions (determined by a greedy policy) are expanded.
    - ``cutoff_depth``: Defines the interval (in number of actions) at which
      the open set is pruned.
    - ``cutoff_size``: The maximum allowed size of the open set before
      triggering a forced prune.
    - ``cutoff_ratio``: The fraction of nodes to keep during a prune operation.
      Pruning retains nodes that are deeper and have better f-scores.
    - ``defense_policy``: A fallback strategy (e.g., beam search completion)
      invoked if the A* search fails to find a complete solution within the
      time limit.

    Workflow:
    1. The search starts from the initial ``partial`` solution (or root).
    2. Nodes are expanded based on the weighted A* priority.
    3. The search tracks the "best partial solution" found so far, prioritizing
       depth first, then cost.
    4. To prevent state explosion, the open set is periodically pruned based
       on ``cutoff_depth`` and ``cutoff_size``. The pruning strategy favors
       deep nodes to drive the search towards completion.
    5. If a complete solution is found, it is recorded. The search continues
       to find better solutions until the open set is empty or time runs out.
    6. If no complete solution is found, the ``defense_policy`` is applied to
       the best partial solution to attempt to generate a valid result.
    """

    start_time = time.time()
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])
    open_heap: List[Tuple[float, AStarNode]] = []  # (f, node)
    open_size = 0


    # //// Push node for ps
    def _push_open(ps: PartialSolution) -> None:
        """Helper: compute priority for ``ps`` and push into ``open_heap``."""
        nonlocal open_heap, open_size

        f, g, h, w = _astar_priority(
            ps,
            eps,
            cost_function,
            pred_function,
            weight_function,
            seed=41*seed if seed is not None else None
        )
        node = AStarNode(partial=ps, g=g, h=h, w=w, f=f)
        heapq.heappush(open_heap, (node.f, node))
        open_size += 1

    _push_open(partial)     # Push root


    # //// Cutoff open heap
    def _cutoff_open() -> None:
        """Helper: cutoff the open heap to the top fraction."""
        nonlocal open_heap, open_size

        open_size = int(open_size * cutoff_ratio)
        open_sorted = sorted(
            open_heap, key=lambda x: (-x[1].partial.num_actions, x[0])
        )
        open_heap = open_sorted[: open_size]
        heapq.heapify(open_heap)


    # //// Main loop
    iterations = 0
    best_solution: Optional[Solution] = None
    best_sol_cost: int = 10**18
    best_partial: PartialSolution = partial
    best_partial_depth: int = partial.num_actions
    best_partial_cost: int = 10**18
    found_complete = False
    status = "done"

    while open_heap:
        if time_limit is not None and time.time() - start_time >= time_limit * 0.95:
            status = "overtime"
            if verbose:
                print(f"[A*] Time limit reached after {iterations} iterations")
            break

        # Logging
        if verbose:
            # Log at 1, 2, ..., 10, 20, ..., 100, 200, ...
            magnitude = 10 ** (len(str(iterations)) - 1)
            if iterations % magnitude == 0 or iterations % 1000 == 0:
                elapsed = time.time() - start_time
                min_f = min(open_heap, key=lambda x: x[0])[0]
                max_f = max(open_heap, key=lambda x: x[0])[0]
                min_depth = min(
                    open_heap,
                    key=lambda x: x[1].partial.num_actions,
                )[1].partial.num_actions
                print(
                    f"[A*] Iteration {iterations}, "
                    f"heap_size={open_size}, " # account for pop
                    f"depth={min_depth} - {best_partial_depth}, "
                    f"best_partial_cost={best_partial_cost}, "
                    f"best_sol_cost={'N/A' if best_sol_cost == 10**18 else best_sol_cost}, "
                    f"best_partial_f={min_f:.2f}, "
                    f"worst_partial_f={max_f:.2f}, "
                    f"time={elapsed:.2f}s",
                )
                # best_partial.stdin_print(verbose=True)
                # print(enumerate_actions_greedily(best_partial, None))

            if best_sol_cost < 10**18 and not found_complete:
                print(
                    f"[A*] Found a complete solution at iteration {iterations}. "
                    f"Cost: {best_sol_cost}"
                )
                found_complete = True

        # Pop best node
        _, node = heapq.heappop(open_heap)
        ps = node.partial

        # Track the depth, cost of the best partial solution
        depth = ps.num_actions
        cost = ps.max_cost
        if depth > best_partial_depth:
            best_partial_depth = depth
            best_partial = ps
            best_partial_cost = cost

            # Cutoff by periodic depth
            if ps.num_actions % cutoff_depth == 0 and ps.num_actions > 0:
                _cutoff_open()

        elif depth == best_partial_depth and cost < best_partial_cost:
            best_partial = ps
            best_partial_cost = cost

        # Cutoff by size
        if open_size >= cutoff_size:
            _cutoff_open()

        # Terminal node: update best complete solution if improved.
        if ps.is_completed():
            sol = ps.to_solution()
            assert sol

            if sol.max_cost < best_sol_cost:
                best_solution = sol
                best_sol_cost = sol.max_cost

            continue

        # Expand children.
        actions = enumerate_actions_greedily(ps, None)
        for action in actions[:width]:
            child_ps = ps.copy()
            apply_general_action(child_ps, action)
            _push_open(child_ps)

        # Update counter
        iterations += 1


    # //// Finalize
    # Summary
    stats: Dict[str, Any] = {
        "iterations": iterations,
        "time": time.time() - start_time,
        "found_complete": best_solution is not None,
        "status": status,
    }

    # Defense policy if no complete solution found
    if best_solution is None:
        if verbose:
            print("[A*] No complete solution found, invoking defense policy...")
        best_solution = defense_policy(best_partial, seed=seed, verbose=verbose)
    assert best_solution

    # Relocate operator refinements
    if verbose:
        print(f"[A*] Applying relocate operator to final solution...")
    best_partial = PartialSolution.from_solution(best_solution)
    refined_partial, _, _ = relocate_operator(
        best_partial,
        mode='first',
        seed=None if seed is None else 4 * seed + 123
    )
    best_solution = refined_partial.to_solution();  assert best_solution
    best_cost = best_solution.max_cost
    if verbose:
        print(
            f"[A*] After relocate, final solution cost: {best_cost}"
        )
    
    if verbose:
        print()
        print(
            f"[A*] Completed."
            f"Best solution cost: {best_solution.max_cost} "
            f"in {stats['time']:.3f}s (iterations={iterations}).",
        )
        print("------------------------------")
        print()
    
    return best_solution, stats




# ====================== Example Usage ==================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    solution, _ = astar_solver(
        problem=test_problem,
        width=5,
        cutoff_depth=5,
        cutoff_size=1000,
        cutoff_ratio=0.286,
        time_limit=60.0,
        seed=42,
        verbose=True,
    )
