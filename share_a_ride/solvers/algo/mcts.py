"""
Monte Carlo Tree Search solver for Share-a-Ride.
"""
import cProfile
import heapq
import math
import time
import random

from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, PartialSolutionSwarm, Solution


# Action format: (taxi_index, action_kind, request_index, incremental_cost)
Action = Tuple[int, str, int, int]
ValueFunction = Callable[[PartialSolution], float]
SelectionPolicy = Callable[[PartialSolution, List[Action]], Optional[Action]]
SimulationPolicy = Callable[[PartialSolution], Optional[PartialSolution]]
DefensePolicy = Callable[[PartialSolution], Optional[Solution]]

FAILED_ROLLOUT_COST = 10**12  # int sentinel for failed rollouts


def _enumerate_actions_greedily(partial: PartialSolution, width: Optional[int]) -> List[Action]:
    """Greedy, beam-style enumeration of next actions.

    This mirrors the pragmatic expansion in beam search:
    - pick a few least-cost taxis,
    - for each, keep a few best incremental moves,
    - optionally allow return-to-depot when route is empty and not at depot.
    """
    assert width is not None and width > 0, "Width must be a positive integer"

    # Collected a greedy set of actions
    available_taxis = [
        idx for idx, state in enumerate(partial.route_states)
        if not state["ended"]
    ]
    if not available_taxis:
        return []
    taxi_order = sorted(
        available_taxis,
        key=lambda idx: partial.route_states[idx]["cost"],
    )
    problem = partial.problem
    taxi_considered = min(
        2 if problem.K >= 50
        else 3 if problem.K >= 20
        else 4,
        len(taxi_order),
    )
    taxi_branches = taxi_order[:taxi_considered]


    # Determine closing depth to delay return-to-depot actions, mirroring beam search.
    # This uses the same total action horizon approximation as the beam search module.
    total_actions = max(1, 2 * (problem.N + problem.M) + problem.K)
    closing_depth = max(0, total_actions - 2 * problem.K)


    # //// Collect actions from selected taxis ////
    actions: List[Action] = []
    for t_idx in taxi_branches:
        # Tracking route state
        state = partial.route_states[t_idx]
        potential_actions = partial.possible_actions(t_idx)
        can_return = (
            state["passenger"] == 0 and
            not state["parcels"] and
            not state["ended"] and
            state["pos"] != 0
        )

        # Collect actions (non-return) for this taxi
        if potential_actions:
            # Keep only a small number of best incremental moves per taxi.
            action_limit = min(
                1 if problem.num_nodes >= 500
                else 2 if problem.num_nodes >= 200
                else 4,
                len(potential_actions),
            )
            limited_actions = sorted(
                potential_actions, key=lambda item: item[2]
            )[:action_limit]

            # Add limited actions
            for kind, node_idx, inc in limited_actions:
                actions.append((t_idx, kind, node_idx, inc))


        # Add return only after closing depth threshold.
        if can_return and partial.n_actions >= closing_depth:
            inc_back = problem.D[state["pos"]][0]
            actions.append((t_idx, "return", 0, inc_back))

    # Filter actions as a defensive measure
    filtered_actions = [
        a for a in actions
        if not (a[1] == "return" and partial.n_actions < closing_depth)
    ]

    # Sort actions globally by incremental cost and apply width limit
    final_actions = sorted(filtered_actions, key=lambda item: item[3])[:width]

    return final_actions


# Apply an action to a partial solution
def _apply_action(partial: PartialSolution, action: Action) -> None:
    """
    Apply ``action`` to ``partial`` solution in-place.
    """
    taxi, kind, node_idx, inc = action
    if kind == "return":
        partial.apply_return_to_depot(taxi)
    else:
        partial.apply_action(taxi, kind, node_idx, inc)





@dataclass
class RewardFunction:
    """
    Tracks observed values during MCTS simulations and maps them to rewards in [0, 1].

    Pipeline intent:
    - A rollout produces a solution/partial with an intrinsic cost profile.
    - ``value_function`` converts that profile into a scalar "value"
      (e.g., negative max-route-cost minus a weighted std of route costs).
    - This class normalizes observed values via min-max into a reward in [0, 1].
      Higher value -> higher reward. Returns 0.5 when insufficient information.
    """
    visits: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")


    def update(self, value: float) -> None:
        """Update the normalization statistics with a new observed ``value``."""
        if not math.isfinite(value):
            return
        self.visits += 1
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)


    def reward_from_value(self, value: float) -> float:
        """
        Map a scalar ``value`` to [0, 1] using min-max normalization over observed values.
        Return 0.5 if no information is available, 0.0 if value is non-finite.
        """
        if not math.isfinite(value):
            return 0.0
        if self.visits == 0:
            return 0.5
        if self.max_value == self.min_value:
            return 0.5
        span = self.max_value - self.min_value
        norm = (value - self.min_value) / span
        return max(0.0, min(1.0, norm))



@dataclass
class MCTSNode:
    """
    Monte Carlo Tree Search node representing a partial solution state.
    Maintains statistics for UCT selection and child nodes.
    """
    partial: PartialSolution  # The partial solution at this node
    parent: Optional["MCTSNode"] = None  # Parent node in the MCTS tree
    action: Optional[Action] = None  # Action taken to reach this node
    width: Optional[int] = None  # Width limit for action enumeration, all actions if None
    children: List["MCTSNode"] = field(default_factory=list)    # Child nodes list
    visits: int = 0  # Number of times this node has been visited
    total_cost: int = 0     # Total accumulated cost from simulations
    total_reward: float = 0.0  # Total accumulated reward from simulations
    untried_actions: List[Action] = field(default_factory=list)  # Actions yet to be tried


    def __post_init__(self) -> None:
        self.untried_actions = _enumerate_actions_greedily(self.partial, self.width)


    @property
    def is_terminal(self) -> bool:
        """
        Check if the node represents a complete solution.
        """
        return self.partial.is_complete()


    @property
    def average_reward(self) -> float:
        """
        Calculate the average reward of this node, which is the total reward
        accumulated from simulations divided by the number of visits.
        """
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


    @property
    def average_cost(self) -> float:
        """
        Calculate the average cost of this node, which is the total cost
        accumulated from simulations divided by the number of visits.
        """
        if self.visits == 0:
            return 0.0
        return self.total_cost / self.visits


    def uct_score(self, uct_c: float) -> float:
        """
        Calculate the UCT score for this node by the standard UCB1 formula:
        UCT = normalized_reward + uct_c * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float("inf")

        exploit_term = self.average_reward

        parent_visits = self.parent.visits if self.parent else self.visits
        explore_term = uct_c * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )

        return exploit_term + explore_term



def _select(root: MCTSNode, exploration: float) -> List[MCTSNode]:
    """
    Perform the selection phase of MCTS, returning the path from root to leaf.
    This should stop at the first node with untried actions or no children.
    """
    path = [root]
    current = root
    while True:
        if current.untried_actions:
            return path
        if not current.children:
            return path
        current = max(current.children, key=lambda child: child.uct_score(exploration))
        path.append(current)



def _expand(
    node: MCTSNode,
    selection_policy: SelectionPolicy,
    width: Optional[int]
) -> Optional[MCTSNode]:
    """
    Expand ``node`` by applying an untried action and returning the considered child.
    """
    # Retrieve untried actions
    if not node.untried_actions:
        node.untried_actions = _enumerate_actions_greedily(node.partial, width)

    # If there are no untried actions, return None
    if not node.untried_actions:
        return None

    # Select an action using the provided selection policy
    action = selection_policy(node.partial, node.untried_actions)
    if action is None:
        return None

    # Remove the selected action from untried actions
    try:
        node.untried_actions.remove(action)
    except ValueError:
        pass

    # Create the new partial solution by applying the action
    child_partial = node.partial.copy()
    _apply_action(child_partial, action)

    # Encapsulate the new partial into a child node
    child = MCTSNode(child_partial, parent=node, action=action, width=width)
    node.children.append(child)

    # Return the newly created child node
    return child



def _backpropagate(path: List[MCTSNode], cost: int, reward: float) -> None:
    """Propagate ``reward`` and raw integer ``cost`` along ``path``."""
    for node in reversed(path):
        node.visits += 1
        node.total_reward += reward
        node.total_cost += cost


def _gather_leaves(
    node: MCTSNode,
    value_function: ValueFunction,
    limit: Optional[int] = None,
) -> List[MCTSNode]:
    """
    Collect leaf nodes, ranking them exclusively with ``value_function``. When
    ``limit`` is provided, only the top scoring leaves are retained.
    """
    if limit is None:
        limit = 10 ** 6  # Large number to gather all leaves

    assert limit is not None and limit > 0, "Limit must be positive"

    # Limited gathering using a min-heap to track top leaves
    heap: List[Tuple[float, int, MCTSNode]] = []
    ticket = count()

    # Depth-first search to collect leaves
    def _collect_limited(current: MCTSNode) -> None:
        # Leaf node, evaluate and consider for heap
        if not current.children:
            score = value_function(current.partial)
            entry = (score, next(ticket), current)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)
            return

        # Recurse into children
        for child in current.children:
            _collect_limited(child)

    _collect_limited(node)

    # Extract ordered leaves from heap
    ordered = sorted(heap, key=lambda item: item[0], reverse=True)
    return [item[2] for item in ordered]



def _run_mcts(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,

    width: Optional[int],
    uct_c: float,
    max_iters: Optional[int],

    seed: Optional[int],
    time_limit: Optional[float],
    verbose: bool,
) -> Tuple[MCTSNode, Optional[PartialSolution], Dict[str, float]]:
    """
    Run the MCTS algorithm with the specified parameters.
    """
    start = time.time()
    reward_function = RewardFunction()

    if seed is not None:
        random.seed(seed)
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    # Initialize the root node
    base_partial = partial or PartialSolution(problem=problem, routes=[])
    root = MCTSNode(base_partial, width=width)

    # Tracker variables
    iterations = 0
    best_rollout: Optional[PartialSolution] = None
    best_rollout_cost = 10 ** 9
    max_depth = 0


    # //////// Main MCTS loop ////////
    while True:
        # Break conditions
        if max_iters is not None and iterations >= max_iters:
            if verbose:
                print(f"[MCTS] Reached max iterations: {max_iters}")
            break
        if time_limit is not None and (time.time() - start) >= time_limit:
            if verbose:
                print(f"[MCTS] Reached time limit: {time_limit:.2f}s")
            break

        # Selection
        path = _select(root, uct_c)
        leaf = path[-1]
        current_depth = len(path) - 1
        if current_depth > max_depth:
            max_depth = current_depth

        # Expansion
        if not leaf.is_terminal:
            child = _expand(leaf, selection_policy, width)
            if child is not None:
                path.append(child)
                working = child
            else:
                working = leaf
        else:
            break   # Terminal node reached; end search

        # Simulation and rollout evaluation (run on a defensive deep copy)
        rollout_result = simulation_policy(working.partial.copy())
        # Successful rollout
        if rollout_result and rollout_result.is_complete():
            rollout_cost = rollout_result.max_cost  # int (objective to minimize)
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_rollout = rollout_result

            # Compute scalar value from the rollout using provided value function
            # (e.g., negative max cost minus weighted std of route costs)
            value = float(value_function(rollout_result))

            # Normalize value to [0,1] reward using observed min/max
            reward_function.update(value)
            reward = reward_function.reward_from_value(value)

        # Failed rollout, use fallback cost and value
        else:
            rollout_cost = FAILED_ROLLOUT_COST  # int
            value = -float(rollout_cost)
            reward_function.update(value)
            reward = reward_function.reward_from_value(value)

        # Backpropagation
        _backpropagate(path, rollout_cost, reward)
        iterations += 1

        # Verbose output every fixed number of iterations
        if verbose and (iterations % 1000 == 0):
            elapsed = time.time() - start
            print(
                f"[MCTS] [Iteration {iterations}] "
                f"Best rollout cost={best_rollout_cost:.3f} "
                f"MaxDepth={max_depth} Time={elapsed:.2f}s"
            )

    # //////// End of MCTS loop ////////


    # Summary info
    info = {
        "iterations": iterations,
        "time": time.time() - start,
        "best_rollout_cost": best_rollout_cost,
    }

    # Verbose output
    if verbose:
        print(
            f"[MCTS] Iterations count={iterations} "
            f"Time={info['time']:.3f}s. Best rollout cost={best_rollout_cost:.3f}"
        )

    return root, best_rollout, info



def mcts_enumerator(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    # Function and policy parameters
    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,

    # Numerical parameters
    best_k: int = 5,
    width: Optional[int] = 5,
    uct_c: float = math.sqrt(2),
    max_iters: Optional[int] = 500,

    # Meta-parameters
    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[List[PartialSolution], Dict[str, float]]:
    """Run MCTS to enumerate top-k partial solutions."""

    tree, _, info = _run_mcts(
        problem,
        partial,
        width=width,
        uct_c=uct_c,
        max_iters=max_iters,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        time_limit=time_limit,
        seed=seed,
        verbose=verbose,
    )

    top_leaves = _gather_leaves(
        tree,
        value_function=value_function,
        limit=max(1, best_k),
    )
    # Extract copy for each partial solution
    top = [leaf.partial.copy() for leaf in top_leaves]

    return top, info



def mcts_solver(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,
    defense_policy: DefensePolicy,

    width: Optional[int] = 5,
    uct_c: float = math.sqrt(2),
    max_iters: Optional[int] = 1000,

    seed: Optional[int] = None,
    time_limit: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, float]]:
    """
    Run a single MCTS search and return the best rollout solution found

    Empirically, enumerating and rescoring leaves did not yield better solutions
    than the direct best rollout tracked during simulations. Therefore
    this solver now:
      1. Runs ``_run_mcts`` to obtain ``best_rollout`` and search stats.
      2. Conver the best rolloout to solution, populating info accordingly and returning it. 
    """
    start = time.time()

    # Run MCTS to get the best rollout
    _, best_rollout, info = _run_mcts(
        problem=problem,
        partial=partial,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        width=width,
        uct_c=uct_c,
        max_iters=max_iters,
        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )

    # Defensive check: No rollout found
    if best_rollout is None:
        info["used_best_rollout"] = False
        info["final_value"] = float("nan")
        return None, info

    # Conver the best rolloout to solution
    sol = best_rollout.to_solution()
    assert sol is not None and sol.is_valid(), "Best rollout is not a valid solution."

    # Populate info
    info["used_best_rollout"] = True
    info["iterations"] = info.get("iterations", 0)
    info["time"] = time.time() - start

    return sol, info



if __name__ == "__main__":
    from share_a_ride.core.utils.generator import generate_instance_coords
    from share_a_ride.solvers.utils.weighter import softmax_weighter
    from share_a_ride.solvers.utils.sampler import sample_from_weight
    from share_a_ride.solvers.algo.greedy import greedy_balanced_solver
    from share_a_ride.solvers.algo.beam import beam_search_solver, parsol_scorer

    # Example usage
    prob = generate_instance_coords(
        N = 40,
        M = 10,
        K = 7,
        area = 1000,
        seed = 60,
    )

    def v_func(
        parsol: PartialSolution,
    ) -> float:
        """Value function provided by negative `parsol_scorer` in beam search"""
        cost = parsol_scorer(parsol)
        return -cost    # IMPORTANT: Use negative cost as value for minimization problems


    def stochastic_select_policy(
        _ps: PartialSolution,
        actions: List[Action],
    ) -> Optional[Action]:
        """Sample an action using a low-temperature softmax over incremental cost."""
        rng = random.Random()
        if not actions:
            return None
        increments = [float(action[3]) for action in actions]
        weights = softmax_weighter(increments, T=0.1)
        chosen_idx = sample_from_weight(rng, weights)
        return actions[chosen_idx]


    def sim_policy(
        ps: PartialSolution,
    ) -> Optional[PartialSolution]:
        """Greedy balanced simulation policy"""
        sim_solution, _ = greedy_balanced_solver(
            ps.problem,
            premature_routes=[r.copy() for r in ps.routes],
            verbose=False,
        )
        return ps if sim_solution is None else PartialSolution.from_solution(sim_solution)


    def def_policy(
        ps: PartialSolution,
    ) -> Optional[Solution]:
        """
        Beam search defense policy.
        """
        beam_solution, _ = beam_search_solver(
            ps.problem,
            cost_function=parsol_scorer,
            initial=PartialSolutionSwarm([ps]),
        )

        return beam_solution


    # //////// Main run ////////
    def _run():
        return mcts_solver(
            problem=prob,
            partial=None,
            value_function=v_func,
            selection_policy=stochastic_select_policy,
            simulation_policy=sim_policy,
            defense_policy=def_policy,

            width=3,
            uct_c=5.0,
            max_iters=20000,

            seed=42,
            time_limit=20.0,
            verbose=False,
        )

    prof = cProfile.Profile()
    prof.enable()
    final_solution, _ = _run()
    prof.disable()
    prof.dump_stats("mcts_profile.prof")


    # Print solution
    assert final_solution is not None
    final_solution.stdin_print(verbose=True)
