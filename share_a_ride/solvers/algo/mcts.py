"""
Monte Carlo Tree Search solver for Share-a-Ride.
Implements MCTS with customizable value, selection, simulation, and defense policies.

There is a note: the main score used in MCTS is value (unnormalized) and reward (normalized [0, 1]).
Value map cost to a higher-better scalar, while reward is value normalized for UCT calculations.
"""
import heapq
import math
import time
import random

from dataclasses import dataclass, field
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple, ParamSpec, Concatenate, Any

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, Solution
from share_a_ride.solvers.algo.greedy import greedy_solver, iterative_greedy_solver
from share_a_ride.solvers.algo.utils import (
    Action, balanced_scorer, enumerate_actions_greedily, apply_general_action
)
from share_a_ride.solvers.operator.relocate import relocate_operator
from share_a_ride.solvers.utils.weighter import softmax_weighter, action_weight
from share_a_ride.solvers.utils.sampler import sample_from_weight




# =============== Type aliases ================
Params = ParamSpec("Params")
ValueFunction = Callable[
    Concatenate[PartialSolution, Params], float
]
SelectionPolicy = Callable[
    Concatenate[List[Action], Params], Optional[Action]
]
SimulationPolicy = Callable[
    Concatenate[PartialSolution, Params],
    Optional[PartialSolution]
]
DefensePolicy = Callable[
    Concatenate[PartialSolution, Params],
    Optional[Solution]
]




# ================ Component functions ================
def _default_vfunc(
        partial: PartialSolution,
        sample_size=6,
        w_std=0.15,
        seed: Optional[int] = None
    ) -> float:
    """Default value function: negative max route cost."""
    return -balanced_scorer(
        partial,
        sample_size=sample_size,
        w_std=w_std,
        seed=seed
    )


def _default_selpolicy(
        actions: List[Action],
        seed: Optional[int] = None,
        t: float = 0.1
    ) -> Optional[Action]:
    """Default selection policy: choose action with minimal incremental cost."""
    rng = random.Random(seed)

    if not actions:
        return None

    weights = softmax_weighter([action_weight(a) for a in actions], t=t)
    chosen_idx = sample_from_weight(rng, weights)

    return actions[chosen_idx]


def _default_simpolicy(
        partial: PartialSolution,
        seed: Optional[int] = None
    ) -> Optional[PartialSolution]:
    """Default simulation policy: greedy balanced solver."""
    sim_solution, _ = greedy_solver(
        partial.problem,
        partial=partial,
    )
    assert sim_solution is not None, "Greedy solver failed in simulation policy."
    seed= 107 * seed + 108 if seed is not None else None

    return PartialSolution.from_solution(sim_solution)


def _default_defpolicy(
        partial: PartialSolution,
        verbose: bool=False,
        seed: Optional[int] = None      # pylint: disable=unused-argument
    ) -> Optional[Solution]:
    """Default defense policy: beam search solver."""
    def_sol, _ = iterative_greedy_solver(
        partial.problem,
        partial=partial,
        iterations=2000,
        time_limit=20.0,
        verbose=verbose,
    )

    return def_sol




# ================ Core logic functions ================
@dataclass
class RewardFunction:
    """
    Tracks observed values during MCTS simulations and maps them to rewards in [0, 1].

    Pipeline intent:
    - A rollout produces a solution/partial with an intrinsic cost profile.
    - ``value_function`` converts that profile into a scalar "value"
        e.g., negative max-route-cost minus a weighted std of route costs).
    - This class normalizes observed values via min-max into a reward in [0, 1].
        Higher value -> higher reward. Returns 0.5 when insufficient information.
    """
    visits: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")


    # Update statistics with new observed value
    def update(self, value: float) -> None:
        """Update the normalization statistics with a new observed ``value``."""
        if not math.isfinite(value):
            return
        self.visits += 1
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)


    # Map value to [0, 1] reward
    def reward_from_value(self, value: float, reward_pow: float = 1.0) -> float:
        """
        Map a scalar ``value`` to [0, 1] using min-max normalization over observed values.
        Apply a power transformation (reward_pow) to sharpen or soften the reward distribution.
        Return 0.5 if no information is available, 0.0 if value is non-finite.
        """
        if not math.isfinite(value):
            return 0.0
        if self.visits == 0:
            return 0.5 ** reward_pow
        if self.max_value == self.min_value:
            return 0.5 ** reward_pow

        # Min-max normalization
        span = self.max_value - self.min_value
        norm = (value - self.min_value) / span

        # Power transformation
        norm = norm ** reward_pow

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


    # Initialize untried actions after node creation
    def __post_init__(self) -> None:
        self.untried_actions = enumerate_actions_greedily(self.partial, self.width)


    # Terminal check
    @property
    def is_terminal(self) -> bool:
        """
        Check if the node represents a complete solution.
        """
        return self.partial.is_completed()


    # Average reward calculation
    @property
    def average_reward(self) -> float:
        """
        Calculate the average reward of this node, which is the total reward
        accumulated from simulations divided by the number of visits.
        """
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


    # Average cost calculation
    @property
    def average_cost(self) -> float:
        """
        Calculate the average cost of this node, which is the total cost
        accumulated from simulations divided by the number of visits.
        """
        if self.visits == 0:
            return 0.0
        return self.total_cost / self.visits


    # UCT score calculation
    def uct_score(self, uct_c: float) -> float:
        """
        Calculate the UCT score for this node by the standard UCB1 formula:
        UCT = normalized_reward + uct_c * sqrt(ln(parent_visits) / visits)
        """
        if self.visits == 0:
            return float("inf")     # Explore all unvisited nodes first

        exploit_term = self.average_reward

        parent_visits = self.parent.visits if self.parent else self.visits
        explore_term = uct_c * math.sqrt(
            math.log(parent_visits + 1) / self.visits
        )

        return exploit_term + explore_term


    # Update from backpropagation
    def update(self, cost: int, reward: float) -> None:
        """
        Update the node's statistics with new cost and reward from a simulation.
        """
        self.visits += 1
        self.total_cost += cost
        self.total_reward += reward




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

        # Select child with highest UCT score
        current = max(
            current.children,
            key=lambda child: child.uct_score(exploration)
        )

        path.append(current)




def _expand(
    node: MCTSNode,
    selection_policy: SelectionPolicy,
    width: Optional[int]
) -> Optional[MCTSNode]:
    """
    Expand ``node`` by applying an untried action and returning the considered child.
    """
    # If there are no untried actions, return None
    if not node.untried_actions:
        return None

    # Select an action using the provided selection policy
    action = selection_policy(node.untried_actions)
    if action is None:
        return None

    # Remove the selected action from untried actions
    try:
        node.untried_actions.remove(action)
    except ValueError:
        pass

    # Create the new partial solution by applying the action
    child_partial = node.partial.copy()
    apply_general_action(child_partial, action)

    # Encapsulate the new partial into a child node
    child = MCTSNode(child_partial, parent=node, action=action, width=width)
    node.children.append(child)

    # Return the newly created child node
    return child




def _backpropagate(path: List[MCTSNode], cost: int, reward: float) -> None:
    """Propagate ``reward`` and raw integer ``cost`` along ``path``."""
    for node in reversed(path):
        node.update(cost, reward)




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
        limit = 10**9  # Large number to gather all leaves

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




# ================ MCTS solver logic ================
def _run_mcts(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution],

    # Numerical parameters
    width: Optional[int],

    # Hyperparameters
    uct_c: float,
    cutoff_depth: int,
    cutoff_depth_inc: int,
    cutoff_iter: int,
    reward_pow: float,

    # Function and policy parameters
    value_function: ValueFunction,
    selection_policy: SelectionPolicy,
    simulation_policy: SimulationPolicy,
    defense_policy: DefensePolicy,

    # Meta-parameters
    seed: Optional[int],
    time_limit: float,
    verbose: bool,
) -> Tuple[MCTSNode, PartialSolution, Optional[Solution], Dict[str, Any]]:
    """
    Run the MCTS algorithm logic. This proceed to iterate over 4 phases of MCTS
    Selection - Expansion - Simulation/Rollout - Backpropagation until a stopping
    criteria is met.

    Return the MCTS tree root, best leaf partial, best solution found, and info dict.
    """
    start = time.time()
    end = start + time_limit
    reward_function = RewardFunction()

    if seed is not None:
        random.seed(seed)
    if partial is None:
        partial = PartialSolution(problem=problem, routes=[])

    # Initialize the root node
    base_partial = partial or PartialSolution(problem=problem, routes=[])
    root = MCTSNode(base_partial, width=width)


    # //////// Main MCTS loop ////////
    iterations = 0
    best_leaf: Optional[PartialSolution] = None
    best_solution: Optional[Solution] = None
    best_solution_cost = 10**18
    max_abs_depth = 0  # Track absolute depth (num_actions) for cutoff decisions
    next_cutoff_depth = cutoff_depth
    cutoff_cnt = 0
    status = "done"
    while True:
        # Break conditions
        if time.time() >= end:
            status = "overtime"
            if verbose:
                print(f"[MCTS] Reached time limit: {time_limit:.2f}s")
            break

        iterations += 1

        # ================ Selection ================
        path = _select(root, uct_c)
        leaf = path[-1]
        abs_depth = leaf.partial.num_actions  # Absolute depth from problem start

        if abs_depth > max_abs_depth:
            max_abs_depth = abs_depth
            if abs_depth > 0 and abs_depth == next_cutoff_depth:
                cutoff_cnt += 1
                next_cutoff_depth += cutoff_depth + cutoff_depth_inc * cutoff_cnt
                root = MCTSNode(leaf.partial, width=width)
                reward_function = RewardFunction()
                if verbose:
                    print(
                        f"[MCTS] Cutoff at iter {iterations}, " 
                        f"abs_depth {abs_depth}, new root set."
                    )

                continue  # Restart iteration with new root

        # Iteration cutoff
        if iterations > 0 and iterations % cutoff_iter == 0:
            root = MCTSNode(leaf.partial, width=width)
            reward_function = RewardFunction()
            if verbose:
                print(
                    f"[MCTS] Cutoff at iteration {iterations}, "
                    f"depth {abs_depth}, new root set."
                )

            continue  # Restart iteration with new root


        # ================ Expansion ================
        if not leaf.is_terminal:
            child = _expand(leaf, selection_policy, width)
            if child is not None:
                path.append(child)
                working = child
            else:
                working = leaf
        else:
            break   # Terminal node reached; end search


        # ================ Simulation and rollout evaluation ===============
        rollout_result = simulation_policy(
            working.partial.copy(), seed=12*seed if seed is not None else None
        )
        if rollout_result is None or not rollout_result.is_completed(): # Failed rollout
            if verbose:
                print(f"[MCTS] Rollout failed or incomplete at iteration {iterations}.")

            reward_function.update(float('-inf') )
            continue

        solution: Optional[Solution] = rollout_result.to_solution()
        assert solution is not None, "Conversion from rollout to solution failed."

        # Update tracking variables for best solution found
        # The important trick here is we favor deeper partials when costs are equal
        # This encourages exploration of longer routes
        solution_cost = solution.max_cost
        if solution_cost <= best_solution_cost:
            if best_leaf is None or working.partial.num_actions > best_leaf.num_actions:
                best_leaf = working.partial.copy()
                best_solution_cost = solution_cost
                best_solution = solution

        # Update reward function
        def_value = value_function(rollout_result)
        reward_function.update(def_value)
        reward = reward_function.reward_from_value(def_value, reward_pow=reward_pow)


        # ================ Backpropagation ================
        _backpropagate(path, solution_cost, reward)

        # Logging
        if verbose:
            # Log at 1, 2, ..., 10, 20, ..., 100, 200, ...
            magnitude = 10 ** (len(str(iterations)) - 1)
            if iterations % magnitude == 0 or iterations % 1000 == 0:
                elapsed = time.time() - start
                print(
                    f"[MCTS] [Iteration {iterations}] "
                    f"Cost: {best_solution_cost:.3f}, "
                    f"Value range: {reward_function.min_value:.3f} "
                    f"- {reward_function.max_value:.3f}, "
                    f"Depth: {abs_depth}, "
                    f"Max depth: {max_abs_depth}, "
                    f"Time: {elapsed:.2f}s.",
                )


    # Summary info
    stats = {
        "iterations": iterations,
        "time": time.time() - start,
        "best_rollout_cost": best_solution_cost,
        "status": status,
    }

    # Verbose output
    if verbose:
        print(
            f"[MCTS] Iterations count: {iterations}, "
            f"Max absolute depth reached: {max_abs_depth}, "
            f"Time={stats['time']:.3f}s."
        )
        print(
            f"[MCTS] Best leaf depth: {best_leaf.num_actions if best_leaf else 'N/A'} "
            f"with rollout cost: {best_solution_cost:.3f}."
        )

    if best_leaf is None:
        best_leaf = root.partial

    # Defensive policy integration
    if best_leaf is not None and best_leaf.is_pending():
        if verbose:
            print(f"[MCTS] Applying defense policy on best leaf...")
        def_sol = defense_policy(
            best_leaf, verbose=verbose, seed=24 * seed if seed is not None else None
        )
        if def_sol is not None:
            cost = def_sol.max_cost
            if best_solution is None or cost < best_solution_cost:
                if verbose:
                    print(f"[MCTS] Defense policy improved solution: {best_solution_cost:.3f} -> {cost:.3f}")
                best_solution = def_sol
                best_solution_cost = cost
                stats["best_rollout_cost"] = best_solution_cost


    return root, best_leaf, best_solution, stats




# ================ MCTS solvers ================
def mcts_enumerator(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution] = None,

    # Numerical parameters
    n_return: int = 5,

    # Numerical parameters
    width: Optional[int] = 3,
    uct_c: float = 0.58,
    cutoff_depth: int = 9,
    cutoff_depth_inc: int = 4,
    cutoff_iter: int = 11300,
    reward_pow: float = 1.69,
    # Function and policy parameters
    value_function: ValueFunction = _default_vfunc,
    selection_policy: SelectionPolicy = _default_selpolicy,
    simulation_policy: SimulationPolicy = _default_simpolicy,
    defense_policy: DefensePolicy = _default_defpolicy,

    # Meta-parameters
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[List[PartialSolution], Dict[str, Any]]:
    """Run MCTS to enumerate top-k partial solutions."""

    tree, _, _, info = _run_mcts(
        problem,
        partial,
        width=width,
        uct_c=uct_c,
        cutoff_depth=cutoff_depth,
        cutoff_depth_inc=cutoff_depth_inc,
        cutoff_iter=cutoff_iter,
        reward_pow=reward_pow,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        defense_policy=defense_policy,
        time_limit=time_limit,
        seed=seed,
        verbose=verbose,
    )

    top_leaves = _gather_leaves(
        tree,
        value_function=value_function,
        limit=max(1, n_return),
    )
    # Extract copy for each partial solution
    top = [leaf.partial.copy() for leaf in top_leaves]

    return top, info




def mcts_solver(
    problem: ShareARideProblem,
    partial: Optional[PartialSolution] = None,

    width: Optional[int] = 3,
    uct_c: float = 0.58,
    cutoff_depth: int = 9,
    cutoff_depth_inc: int = 4,
    cutoff_iter: int = 11300,
    reward_pow: float = 1.69,
    value_function: ValueFunction = _default_vfunc,
    selection_policy: SelectionPolicy = _default_selpolicy,
    simulation_policy: SimulationPolicy = _default_simpolicy,
    defense_policy: DefensePolicy = _default_defpolicy,

    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Run a single MCTS search and return the best rollout solution found

    Empirically, enumerating and rescoring leaves did not yield better solutions
    than the direct best rollout tracked during simulations. Therefore
    this solver now:
        1. Runs ``_run_mcts`` to obtain ``best_rollout`` and search stats.
        2. Convert the best rollout to solution, populating info accordingly and returning it.
    """
    start = time.time()

    # Run MCTS to get the best rollout
    _, best_leaf, sol, info = _run_mcts(
        problem=problem,
        partial=partial,
        value_function=value_function,
        selection_policy=selection_policy,
        simulation_policy=simulation_policy,
        defense_policy=defense_policy,
        width=width,
        uct_c=uct_c,
        cutoff_depth=cutoff_depth,
        cutoff_depth_inc=cutoff_depth_inc,
        cutoff_iter=cutoff_iter,
        reward_pow=reward_pow,
        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )
    assert sol

    # Relocate operator refinements
    if verbose:
        print(f"[MCTS] Applying relocate operator to final solution...")
    best_partial = PartialSolution.from_solution(sol)
    refined_partial, _, _ = relocate_operator(
        best_partial,
        mode='first',
        seed=None if seed is None else 4 * seed + 123
    )
    sol = refined_partial.to_solution();  assert sol
    best_cost = sol.max_cost
    if verbose:
        print(
            f"[MCTS] After relocate, final solution cost: {best_cost}"
        )

    # Info population
    info["used_best_rollout"] = True
    info["iterations"] = info.get("iterations", 0)
    info['time'] = time.time() - start

    # Logging
    if verbose:
        if sol is not None:
            print()
            print(
                f"[MCTS] Final solution cost: {sol.max_cost:.3f} "
                f"after {info['iterations']} iterations "
                f"in {info['time']:.2f}s."
            )
            print("------------------------------")
            print()
        else:
            print()
            print(
                f"[MCTS] No solution found after "
                f"{info['iterations']} iterations "
                f"in {info['time']:.2f}s."
            )
            print("------------------------------")
            print()

    return sol, info




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    final_sol, final_info = mcts_solver(
        problem=test_problem,
        width=2,
        cutoff_iter=10000,
        cutoff_depth=10,
        cutoff_depth_inc=2,
        reward_pow=2.0,
        time_limit=60.0,
        seed=123,
        verbose=True,
    )
