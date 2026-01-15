"""
Ant Colony Optimization (ACO) solver for Share-a-Ride.

This module implements the ACO algorithm for solving the Share-a-Ride problem,
which is a Min-Max VRP variant with pickup-and-delivery constraints.
"""
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, ParamSpec, Concatenate, Union, Set

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import PartialSolution, PartialSolutionSwarm, Solution
from share_a_ride.solvers.algo.greedy import greedy_solver, iterative_greedy_solver
from share_a_ride.solvers.algo.utils import enumerate_actions_greedily, apply_general_action, Action
from share_a_ride.solvers.operator.relocate import relocate_operator
from share_a_ride.solvers.utils.weighter import weighted, action_weight
from share_a_ride.solvers.utils.sampler import sample_from_weight




# ================ Type aliases ================
Params = ParamSpec("Params")
ActionNode = Tuple[int, int]
ValueFunction = Callable[Concatenate[PartialSolution, Params], float]
FinalizePolicy = Callable[Concatenate[PartialSolution, Params], Optional[Solution]]




# ================ ACO Policies Implementation ================
def _default_value_function(
        partial: PartialSolution,
        perturbed_samples: int = 6,     # perturb
        seed: Optional[int] = None,     # pylint: disable=unused-argument
    ) -> float:
    _, stats = _, stats = iterative_greedy_solver(
        partial.problem, partial, iterations=perturbed_samples, time_limit=0.1, seed=seed
    )

    return stats["average_cost"]


def _default_finalize_policy(
        partial: PartialSolution,
        seed: Optional[int] = None,
    ) -> Optional[Solution]:
    sol, _info = iterative_greedy_solver(
        partial.problem,
        partial,
        iterations=3000,
        time_limit=3.0,
        seed=seed,
        verbose=False
    )

    if not sol:
        return None

    raw_sol = PartialSolution.from_solution(sol)
    refined_sol, _modified, _cnt = relocate_operator(
        partial=raw_sol,
        seed=seed,
        verbose=False
    )
    sol = refined_sol.to_solution()

    return sol




# ================ ACO Components Implementation ================
@dataclass
class SolutionTracker:
    """
    Tracks and evaluates solution quality (lfunc) for the ACO population.

    Maintains statistics about the best, worst, and average solutions
    found during the optimization process.
    """
    best_solution: Optional[Solution] = None
    best_cost: int = 10**18
    worst_cost: int = -1
    total_cost: int = 0
    count: int = 0


    def update(
            self,
            source: Union[Solution, PartialSolutionSwarm, List[Optional[Solution]]]
        ) -> None:
        """
        Update metrics with a solution or a swarm of partial solutions.
        
        Args:
            source: Either a Solution or a PartialSolutionSwarm to update from.
        """
        if isinstance(source, Solution):
            self._update_from_solution(source)
        elif isinstance(source, PartialSolutionSwarm):
            self._update_from_swarm(source)
        elif isinstance(source, list):
            self._update_from_list(source)


    def _update_from_solution(self, solution: Solution) -> None:
        """Update metrics with a single solution."""
        cost = solution.max_cost

        self.count += 1
        self.total_cost += cost
        self.worst_cost = max(self.worst_cost, cost)

        if cost < self.best_cost:
            self.best_cost = cost
            self.best_solution = solution


    def _update_from_swarm(self, swarm: PartialSolutionSwarm) -> None:
        """Update metrics with a swarm of partial solutions."""
        for partial in swarm.partial_lists:
            if not partial.is_completed():
                continue

            sol = partial.to_solution()
            if not sol:
                continue

            self._update_from_solution(sol)


    def _update_from_list(self, solutions: List[Optional[Solution]]) -> None:
        """Update metrics with a list of solutions."""
        for sol in solutions:
            if sol is None:
                continue

            self._update_from_solution(sol)


    def stats(self) -> Dict[str, float]:
        """Return statistics of the population."""
        avg_cost = self.total_cost / self.count if self.count > 0 else 0.0
        return {
            "best_cost": self.best_cost,
            "worst_cost": self.worst_cost,
            "avg_cost": avg_cost,
            "count": self.count,
        }


    def opt(self) -> Optional[Solution]:
        """Return the best solution found so far."""
        return self.best_solution


class PheromoneMatrix:
    """
    Pheromone matrix (tau) that maintains pheromone levels on edges.
    It has a clamp mechanism to avoid pheromone explosion or vanishing.
    
    Unit: d^(-1)
    """

    def __init__(
            self,
            problem: ShareARideProblem,
            sigma: int,
            rho: float,
            init_cost: int
        ) -> None:

        """Initialize pheromone matrix and parameters."""

        self.size = problem.num_nodes
        self.sigma = sigma
        self.rho = rho
        self.tau_0 = 1.0 / (rho * init_cost)
        self.tau_max = 2 * self.tau_0
        self.tau_min = self.tau_0 / 10.0
        self.tau = [
            [self.tau_0 for _ in range(self.size)]
            for _ in range(self.size)
        ]

        assert 0.0 < self.rho < 1.0, "Evaporation rate rho must be in (0,1)."
        assert sigma >= 1, "Number of elitists sigma must be at least 1."


    def get(self, prev: ActionNode, curr: ActionNode) -> float:
        """Get pheromone level on transition from prev action to curr action."""
        return self.tau[prev[1]][curr[0]]


    def update(
            self,
            swarm: PartialSolutionSwarm,
            opt: Optional[Solution],
        ) -> None:
        """
        Update pheromone on all edges based on the rank-based elitist heuristic
        by Bullnheimer et al. (1999).

        The formula is:
            tau(i,j) = rho * tau(i,j)
                     + sum_{r=1}^{sigma-1} (sigma - r) * d_r(i,j)
                     + sigma * d_best(i,j)

        Where:
        - rho is the evaporation keeping rate
        - sigma is the number of elitists (including the best-so-far solution)
        - d_r(i,j) = 1 / L_r if edge (i,j) is in the r-th ranked solution
        - d_best(i,j) = 1 / L_best if edge (i,j) is in the best-so-far solution
        """

        # //// Helper function to extract edges from a PartialSolution
        def extract_edges(partial: PartialSolution) -> List[Tuple[int, int]]:
            """Extract all (prev_out, curr_in) edges from a partial solution's routes."""
            edges: List[Tuple[int, int]] = []

            for route_idx, _ in enumerate(partial.routes):
                actnodes = partial.enumerate_action_nodes(route_idx)
                for prev, nxt in zip(actnodes[:-1], actnodes[1:]):
                    edges.append((prev[1], nxt[0]))

            return edges


        # //// Sort partials by max_cost (ascending) and take elitists
        ranked_partials = sorted(
            ((par.max_cost, par) for par in swarm.partial_lists),
            key=lambda x: x[0]
        )[:self.sigma]


        # //// Update pheromone matrix
        # Evaporation and clamp to tau_min
        self.tau = [
            [max(self.tau_min, self.rho * val) for val in row]
            for row in self.tau
        ]

        increased_edges: Set[Tuple[int, int]] = set()

        # Add ranked solution contributions: (sigma - rank) * (1 / L_r)
        for rank, (cost, partial) in enumerate(ranked_partials):
            elitist_weight = (self.sigma - rank) / cost

            # Loop through edges
            for (i, j) in extract_edges(partial):
                increased_edges.add((i, j))
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.tau[i][j] += elitist_weight

        # Add best-so-far solution contribution: sigma * (1 / L_best)
        if opt is not None and opt.max_cost > 0:
            best_weight = self.sigma / opt.max_cost

            # Temporarily convert best solution to PartialSolution
            best_partial = PartialSolution.from_solution(opt)

            # Loop through edges
            for (i, j) in extract_edges(best_partial):
                increased_edges.add((i, j))
                if 0 <= i < self.size and 0 <= j < self.size:
                    self.tau[i][j] += best_weight

        # Clamp increased edges to tau_max
        for (i, j) in increased_edges:
            self.tau[i][j] = min(self.tau_max, self.tau[i][j])



class DesirabilityMatrix:
    """
    Desirability/Visibility matrix.

    The two heuristic integrated here are Clark-Wright Savings and
    exponential remaining load ratio.

    The formula is:
        eta_ij = saving_ij^phi * (1/d_ij)^chi * (2 - is_ppick)
                * ( 1 + gamma * ((Q - new_cap) / Q))^kappa
    with:
        - saving_ij = max(0, D[i][0] + D[0][j] - D[i][j])
        - is_ppick = 1 if node j is a passenger pickup, else 0
        - Q = capacity of the taxi
        - sum(q) = current load after reaching node i
    
    The matrix is initialized only once on distance matrix D and remain static
    during ACO iterations. Higher values indicate more desirable edges to traverse.
    """

    def __init__(
            self,
            problem: ShareARideProblem,
            phi: float,
            chi: float,
            gamma: float,
            kappa: float
        ) -> None:
        """
        Initialize desirability matrix using Clark-Wright Savings:
        saving(i,j) = max(0, D[i][0] + D[0][j] - D[i][j])
        eta(i,j) = saving(i,j)^phi * D[i][j]^(-chi)
        This rewards edges that "save" distance compared to going through depot.
        """
        self.size = problem.num_nodes
        self.problem = problem
        self.phi = phi
        self.chi = chi
        self.gamma = gamma
        self.kappa = kappa

        # Initialize saving matrix
        self.saving_matrix: List[List[float]] = []
        D = self.problem.D      # pylint: disable=C0103
        for i in range(self.size):
            row = []

            for j in range(self.size):
                if i == j:
                    row.append(0)   # Dummy
                else:
                    # Clark-Wright savings: positive when combining saves distance
                    slack_ij = max(D[0][i] + D[j][0] - D[i][j], 0)
                    # Add 1 to savings to ensure non-zero desirability even with no savings
                    saving_term = (1 + slack_ij) ** self.phi

                    row.append(saving_term)

            self.saving_matrix.append(row)


    def get(
            self,
            prev: ActionNode,
            curr: ActionNode,
            partial: PartialSolution,
            action: Action
        ) -> float:
        """
        Get desirability value for edge (i, j).
        Multiply two new term:
        - people term: 2 - is_ppick
        - parcel term: (1 + gamma * (Q - new_cap) / Q) * kappa
        """
        # Extract variables
        route_idx, kind, node, inc = action
        Q = self.problem.Q[route_idx]        # pylint: disable=C0103
        state = partial.states[route_idx]
        new_cap = state["load"]
        if kind == "pickL":
            new_cap += self.problem.q[node - 1]
        if kind == "dropL":
            new_cap -= self.problem.q[node - 1]

        # Formulate terms
        distance_term = (1 + weighted(kind, inc)) ** self.chi
        saving_term = self.saving_matrix[prev[1]][curr[0]]
        people_term = 2 - int(kind == "pickP")
        parcel_term = (1 + self.gamma * (Q - new_cap) / Q) * self.kappa

        return (saving_term / distance_term) * people_term * parcel_term


class NearestExpansionCache:
    """
    Additional cache to store nearest neighbor expansions for each nodes in
    the ShareARideProblem instance.

    This is used to speed up lookup for Ant agents.
    """

    def __init__(
            self,
            problem: ShareARideProblem,
            n_nearest: int = 3
        ) -> None:
        """
        Initialize nearest expansion cache for each node.
        """
        self.nearest_actions: List[List[Tuple[str, int, int]]] = []


        # //// Build a surrogate PartialSolution to query possible expansions
        for nodeid in range(problem.num_nodes):
            # For depot (node 0), use [[0], ...] to avoid marking taxi as ended
            # For other nodes, use [[0, node_idx], ...] to simulate being at that node
            if nodeid == 0:
                routes = [[0] for _ in range(problem.K)]
            elif problem.is_ppick(nodeid):
                # Decisions are never made at pickup nodes for passengers
                self.nearest_actions.append([])
                continue
            elif problem.is_pdrop(nodeid):
                pid = problem.rev_pdrop(nodeid)
                pick = problem.pserve(pid)[0]
                routes = [[0, pick, nodeid]] + [[0] for _ in range(problem.K - 1)]
            elif problem.is_lpick(nodeid):
                routes = [[0, nodeid]] + [[0] for _ in range(problem.K - 1)]
            elif problem.is_ldrop(nodeid):
                lid = problem.rev_ldrop(nodeid)
                pick = problem.lpick(lid)
                routes = [[0, pick, nodeid]] + [[0] for _ in range(problem.K - 1)]
            else:
                print(f"[ACO] [Error] Cache error: Unknown node type for node {nodeid}.")
                self.nearest_actions.append([])
                continue

            partial = PartialSolution(problem, routes=routes)


            # //// Get possible expansions and keep nearest num_nearest
            t_acts = partial.possible_expand(0)
            t_acts.sort(key=lambda item: weighted(item[0], item[2]))
            t_acts = t_acts[:n_nearest]

            # Store in cache
            self.nearest_actions.append(t_acts)


    def query(self, partial: PartialSolution, n_queried: int) -> List[Action]:
        """
        Get best cached expansions (non-depot) for the partial, filtered and prioritized like
        enumerate_actions_greedily: balanced first, then by inc cost.

        Returns a list of (weight, Action) tuples.
        """
        if partial.num_actions < partial.problem.num_expansions:
            return []   # No expansions when every request is served (only returns)


        # //// Collect expansion actions ////
        current_max = partial.max_cost
        prioritized: List[Tuple[float, Action]] = []
        secondary: List[Tuple[float, Action]] = []

        for route_idx, state in enumerate(partial.states):
            if state["ended"]:
                continue
            pos = state["pos"]
            cached: List[Tuple[str, int, int]] = self.nearest_actions[pos]

            # Filter and prioritize expansion actions
            for unassigned_action in cached:
                # Extract action components for prioritization
                kind, node_idx, inc = unassigned_action

                # Check validity
                if not partial.check_expand(route_idx, kind, node_idx):
                    continue

                # Reassign taxi for the action
                action: Action = (route_idx, kind, node_idx, inc)
                weight = weighted(kind, inc)

                # Prioritize like enumerate_actions_greedily
                if partial.route_costs[route_idx] + inc <= current_max:
                    prioritized.append((weight, action))
                else:
                    secondary.append((weight, action))

        # Sort by weight
        prioritized.sort(key=lambda x: x[0])
        secondary.sort(key=lambda x: x[0])

        # Combine and return top num_queried actions
        all_actions = [action for _, action in prioritized + secondary]

        return all_actions[:n_queried]


class Ant:
    """
    Ant agent that constructs solutions by traversing the graph.
    
    Each ant maintains its current partial solution and can expand
    routes using pheromone and desirability information.
    """

    class ProbaExpandSampler:
        """
        Probabilistic expansion strategy component for ACO ants.
        It internally calculates the transition probabilities and
        samples actions accordingly.

        Uses NearestExpansionCache to prioritize cached actions before
        falling back to enumerate_actions_greedily for remaining width.
        """
        partial: PartialSolution
        cache: "NearestExpansionCache"
        alpha: float
        beta: float
        omega: float
        q_prob: float
        width: int

        def __init__(
                self,
                partial: PartialSolution,
                cache: "NearestExpansionCache",
                alpha: float,
                beta: float,
                omega: float,
                q_prob: float,
                width: int
            ) -> None:
            self.partial = partial
            self.cache = cache
            self.alpha = alpha
            self.beta = beta
            self.omega = omega
            self.q_prob = q_prob
            self.width = width


        def _get_action_node(self, action: Action) -> ActionNode:
            """Get ActionNode for a given action."""
            _route_idx, kind, actid, _ = action
            prob = self.partial.problem
            if kind == "serveP":
                return prob.pserve(actid)
            elif kind == "pickL":
                node = prob.lpick(actid)
                return (node, node)
            elif kind == "dropL":
                node = prob.ldrop(actid)
                return (node, node)
            else:
                return (0, 0)


        def _collect_actions(self) -> List[Tuple[float, Action]]:
            """
            Collect actions using cache-first strategy. Then reweight them
            to prioritize prioritized actions over secondary ones.
            """
            partial = self.partial
            width = self.width


            # //// Ending case
            if partial.num_actions >= partial.problem.num_expansions:
                actions = enumerate_actions_greedily(partial, width)
                first_weight = action_weight(actions[0])
                return [(first_weight / action_weight(action), action) for action in actions]


            # //// Expansion case
            # First try to get from cache
            actions = self.cache.query(partial, width)

            # If not enough, fill with greedy enumeration
            if len(actions) < width:
                actions = enumerate_actions_greedily(partial, width)[:width]

            # Now we reweight actions based on priority
            # Note that the first secondary action has incremental cost
            # smaller than the last prioritized action
            # Because of that, the trick is to raise the weight of secondary
            # actions above the max weight of prioritized actions
            weight_actions: List[Tuple[float, Action]] = []
            found_secondary = False
            reweight_base = 0.0
            first_action = actions[0]
            first_weight = action_weight(first_action)
            weight_actions.append((first_weight, first_action))

            for prev_action, curr_action in zip(actions[:-1], actions[1:]):
                prev_inc = prev_action[3]
                curr_inc = curr_action[3]

                if not found_secondary and curr_inc > prev_inc:
                    found_secondary = True

                    max_prior_weight = action_weight(curr_action)
                    reweight_base = max_prior_weight + 1.0

                weight = action_weight(curr_action)
                if found_secondary:
                    weight += reweight_base

                weight_actions.append((weight, curr_action))

            # Finally, invert and normalize weights by first action's inc cost
            fitted_actions = [
                (first_weight / weight, action) for weight, action in weight_actions
            ]

            return fitted_actions


        def _compute_log_proba(
                self,
                tau: PheromoneMatrix,
                eta: DesirabilityMatrix,
                fit: float,
                action: Action,
            ) -> float:
            """
            Compute log of transition probability component for edge (from_node, to_node).
            Uses log-space to avoid numerical underflow.
            
            log(p) = alpha * log(tau) + beta * log(eta) + omega * log(fit)
            """
            # Extract from_node and to_node
            route_idx = action[0]
            state = self.partial.states[route_idx]
            prev_out = state["pos"]
            prev_node: ActionNode = (10**18, prev_out)   # Dummy prev node, we only need phys_out
            curr_node: ActionNode = self._get_action_node(action)

            # Get tau and eta values
            tau_val = tau.get(prev_node, curr_node)
            eta_val = eta.get(prev_node, curr_node, self.partial, action)

            # Clamp to avoid log(0)
            tau_val = max(tau_val, 1e-300)
            eta_val = max(eta_val, 1e-300)

            log_proba = (
                + self.alpha * math.log(tau_val)
                + self.beta * math.log(eta_val)
                + self.omega * math.log(fit)
            )

            return log_proba


        def sample_action(
                self,
                tau: PheromoneMatrix,
                eta: DesirabilityMatrix,
                rng: random.Random,
            ) -> Optional[Action]:
            """
            Sample an action using exploitation/exploration strategy.
            
            First collects actions (prioritizing cache), then:
            - With probability q, exploit (choose greedy nearest action).
            - Otherwise, explore by sampling from pheromone-desirability distribution.
            
            Uses log-sum-exp trick for numerical stability.
            
            Returns the selected action (type Action), or None if no actions available.
            """

            # Collect actions using cache-first strategy
            actions = self._collect_actions()
            if not actions:
                return None

            # Compute log probabilities for each action
            log_probas: List[float] = []
            for weight, action in actions:
                log_proba = self._compute_log_proba(tau, eta, weight, action)
                log_probas.append(log_proba)

            # Convert to probabilities using log-sum-exp trick for numerical stability
            # p_i = exp(log_p_i - max_log_p) / sum(exp(log_p_j - max_log_p))
            max_log = max(log_probas)
            exp_shifted = [math.exp(lp - max_log) for lp in log_probas]
            total = sum(exp_shifted)
            probas = [e / total for e in exp_shifted]

            # Select action: exploit vs explore
            select_idx: int
            if rng.random() < self.q_prob:
                # Exploitation: choose action with max probability
                select_idx = probas.index(max(probas))
            else:
                # Exploration: sample from distribution
                select_idx = sample_from_weight(rng, probas)

            return actions[select_idx][1]


    def __init__(
            self,
            partial: PartialSolution,
            cache: "NearestExpansionCache",
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            alpha: float,
            beta: float,
            omega: float,
            q_prob: float,
            width: int,
            rng: random.Random,
        ) -> None:

        """Initialize ant with parameters and partial solution."""
        self.problem = partial.problem
        self.partial = partial
        self.cache = cache
        self.tau = tau
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.q_prob = q_prob
        self.width = width
        self.rng = rng

        # Initialize probabilistic expansion strategy
        self.sampler = Ant.ProbaExpandSampler(
            partial=self.partial,
            cache=cache,
            alpha=alpha,
            beta=beta,
            omega=omega,
            q_prob=q_prob,
            width=width
        )


    def expand(self) -> bool:
        """
        Expand the route construction by selecting and applying the next action.
        
        Delegates action sampling to ProbaExpand component, then applies the
        selected action and records the edge for trail update.
        
        Returns True if expansion was successful, False if complete.
        """
        if self.partial.is_completed():
            return False

        # Sample action using cache-first strategy
        sampled_action = self.sampler.sample_action(
            self.tau, self.eta, self.rng
        )
        if not sampled_action:
            return False

        # Commit action
        apply_general_action(self.partial, sampled_action)

        return True


class AntPopulation:
    """
    The main Ant population manager class for ACO.
    Handle two phases of an iteration:
    - Construction: each ant builds a complete solution.
    - Pheromone Update: update pheromone trails based on solutions and evaporation rate.
    """

    def __init__(
            self,
            swarm: PartialSolutionSwarm,

            # Cache, Pheromone and desirability matrices (initialized earlier)
            cache: NearestExpansionCache,
            tau: PheromoneMatrix,
            eta: DesirabilityMatrix,
            lfunc: SolutionTracker,

            # Ant params
            alpha: float,
            beta: float,
            omega: float,
            q_prob: float,
            width: int,

            # Run parameters
            depth: int,
            time_limit: float,
            seed: Optional[int],
            verbose: bool
        ) -> None:
        self.swarm = swarm.copy()
        self.completed = [par.is_completed() for par in self.swarm.partial_lists]

        self.cache = cache
        self.tau = tau
        self.eta = eta
        self.lfunc = lfunc

        self.depth = depth
        self.time_limit = time_limit
        self.seed = seed
        self.verbose = verbose

        # Initialize ants
        self.ants: List[Ant] = []
        for idx, partial in enumerate(self.swarm.partial_lists):
            ant = Ant(
                partial=partial,
                cache=cache,
                tau=self.tau,
                eta=self.eta,
                alpha=alpha,
                beta=beta,
                omega=omega,
                q_prob=q_prob,
                width=width,
                rng=random.Random(hash(seed + 100 * idx) if seed else None),
            )
            self.ants.append(ant)

        # Miscellaneous utilities
        self.num_ants = len(self.ants)
        self.max_actions = swarm.problem.num_actions
        self.start_time = time.time()
        self.end_time = self.start_time + self.time_limit
        self.tle = lambda: time.time() > self.end_time


    def expand(self) -> bool:
        """
        Let each ant in the population expand its partial solution
        until completion or no further expansion is possible.
        """
        is_expanded = False
        for idx, ant in enumerate(self.ants):
            if self.completed[idx]:
                continue

            if ant.expand():
                is_expanded = True
            elif self.verbose:
                print(
                    f"[ACO] [Depth {ant.partial.num_actions}] "
                    f"[Warning] Ant {idx + 1} cannot expand, "
                    "further diagnosis needed."
                )
                raise RuntimeError("Ant expansion failure.")

        return is_expanded


    def update(self) -> None:
        """
        Update the pheromone matrix based on the solutions constructed
        """
        self.lfunc.update(
            source=self.swarm,  # Update from the current swarm
        )

        self.tau.update(
            swarm=self.swarm,
            opt=self.lfunc.opt(),
        )


    def run(self) -> PartialSolutionSwarm:
        """
        Run the ACO process by iterating through the following steps:
        - Each ant constructs a complete solution.
        - Update pheromone trails based on the solutions found.
        """
        anchors = [
            self.max_actions // 5,
            self.max_actions // 2,
            self.max_actions * 9 // 10
        ]

        # Loop
        for ite in range(self.depth):
            if self.tle():
                if self.verbose:
                    print("[ACO] Time limit reached, skipping this iteration.")
                return self.swarm

            # Logging
            if self.verbose:
                if ite in anchors or ite % 100 == 0:
                    costs = [par.max_cost for par in self.swarm.partial_lists]
                    depths = [par.num_actions for par in self.swarm.partial_lists]
                    print(
                        f"[ACO] [Iteration {ite}] "
                        f"Partial cost range: {min(costs):.3f} - {max(costs):.3f}, "
                        f"Depth range: {min(depths)} - {max(depths)}, "
                        f"Time_elapsed={time.time() - self.start_time:.2f}s."
                    )

            # Perform expansion and pheromone update
            if not self.expand():
                if self.verbose:
                    print("[ACO] All ants have completed their solutions.")
                break

            self.update()


        # Final logging
        if self.verbose:
            num_sol = sum(1 for par in self.swarm.partial_lists if par.is_completed())
            run_opt = self.swarm.opt()
            global_opt = self.lfunc.opt()
            print(
                f"[ACO] Finished all depth.\n"
                f"Complete solutions found: {num_sol}/{self.num_ants}.\n"
                f"Run best cost: {run_opt.max_cost if run_opt else 'N/A'}, "
                f"Opt cost: {global_opt.max_cost if global_opt else 'N/A'}."
            )

        return self.swarm   # A new swarm with updated partials


class SwarmTracker:
    """
    Tracks and records the best partial for each initial partial across ACO iterations.
    We assume all fed swarms are generated from the AntPopulation from a single
    initial swarm, hence partials are compared accordingly to its index
    (which mean we take the Pareto-optimal swarm)
        Here we need a value function to evaluate the potential of each partial.
    """
    def __init__(
            self,
            swarm: PartialSolutionSwarm,
            value_function: ValueFunction,
            finalize_policy: FinalizePolicy,
            seed: Optional[int] = None,
        ) -> None:

        # Initialization fields
        self.seed = seed

        # Tracking fields
        self.frontier_swarm: List[PartialSolution] = [
            partial.copy() for partial in swarm.partial_lists
        ]
        self.num_partials = swarm.num_partials

        # Fitness fields (value_function now returns float fitness values)
        self.frontier_fitness: List[float] = [
            value_function(partial.copy())
            for partial in self.frontier_swarm
        ]

        # Finalized solutions
        self.finals: List[Solution]
        self.is_finalized: bool = False

        # Policy fields
        self.value_function = value_function
        self.finalize_policy = finalize_policy


    def update(self, source: PartialSolutionSwarm) -> List[float]:
        """
        Update the best partials based on the new swarm.
        Also updates fitness values by applying the value function to improved partials.
        """
        assert self.num_partials == source.num_partials

        for idx, partial in enumerate(source.partial_lists):
            fitness = self.value_function(
                partial.copy(), seed=self.seed + 10 * idx if self.seed else None
            )

            # Compare to the current frontier partial (lower fitness is better)
            if fitness < self.frontier_fitness[idx]:
                self.frontier_swarm[idx] = partial.copy()
                self.frontier_fitness[idx] = fitness

        return self.frontier_fitness


    def finalize(self, cutoff: Optional[int]) -> List[Solution]:
        """
        Finalize the elitists into complete solutions.
        Only finalizes up to `cutoff` partials to save time. Partials are sorted
        by fitness values and the top ones are finalized first.
        
        Args:
            cutoff: Maximum number of partials to finalize (time-saving limit).
                    Should be >= desired return count to avoid missing good solutions.
            time_limit: Remaining time in seconds for finalization.
        
        Returns:
            List of finalized solutions sorted by max_cost.
        """
        # //// Select partials
        sorted_partials = sorted(
            zip(self.frontier_swarm, self.frontier_fitness),
            key=lambda x: x[1]
        )
        chosen_partials = sorted_partials[:cutoff] if cutoff else sorted_partials


        # //// Finalizing
        finalized: List[Solution] = []
        for idx, (par, _fitness) in enumerate(chosen_partials):
            sol = self.finalize_policy(
                par, seed=self.seed + 20 * idx if self.seed else None
            )
            if sol:
                finalized.append(sol)


        # /// Update and return
        finalized.sort(key=lambda s: s.max_cost)
        finalized = finalized[:cutoff] if cutoff else finalized

        # Update final state
        self.is_finalized = True
        self.finals = finalized

        # Return finalized solutions
        return finalized


    def top(
            self,
            k: int,
            cutoff: Optional[int] = None,
        ) -> List[Solution]:
        """
        Return the top-k finalized solutions based on max_cost.
        If not finalized yet, finalize first with the given cutoff and time_limit.
        
        Args:
            k: Number of solutions to return.
            cutoff: Maximum partials to finalize (defaults to k if None).
            time_limit: Remaining time for finalization.
        """
        if cutoff is None:
            cutoff = k

        if not self.is_finalized:
            self.finalize(cutoff)

        return self.finals[:k]


    def opt(
            self,
            cutoff: Optional[int] = None,
        ) -> Solution:
        """
        Return the best solution among the finalized solutions.
        If not finalized yet, finalize first with the given cutoff and time_limit.
        
        Args:
            cutoff: Maximum partials to finalize (defaults to None, but suggested to be 10).
            time_limit: Remaining time for finalization.
        """
        if not self.is_finalized:
            self.finalize(cutoff)

        assert self.finals, "No finalized solutions available."

        return self.finals[0]  # Already sorted by max_cost




# ================ ACO Logic Implementation ================
def _run_aco(
    problem: ShareARideProblem,
    swarm: PartialSolutionSwarm,

    # Run parameters
    n_cutoff: Optional[int],
    iterations: int,
    depth: Optional[int],

    # Hyperparameters
    # / Ants
    q_prob: float,
    alpha: float,
    beta: float,
    omega: float,

    # / Desirability
    phi: float,
    chi: float,
    gamma: float,
    kappa: float,
    # / Pheromone
    sigma: int,
    rho: float,

    # / Width
    width: int,

    # Policies
    value_function: ValueFunction,
    finalize_policy: FinalizePolicy,

    # Meta parameters
    seed: Optional[int],
    time_limit: float,
    verbose: bool,
) -> Tuple[SwarmTracker, Dict[str, Any]]:
    """
    Run the ACO algorithm with the specified parameters for ``depth`` times
    or until ``time_limit`` is reached.
    
    This is the inner function that performs the core ACO logic for one run.
    It initializes all ACO components internally and uses the AntPopulation
    class to manage the ant colony. Uses SwarmTracker to track best partials
    across depth and finalize them into complete solutions.
    
    Args:
        - problem: ShareARideProblem instance.
        - swarm: Initial PartialSolutionSwarm (each partial becomes an ant).
        - iterations: Number of ACO iterations to run.
        - depth: Number of depth/actions to run.
        - q_prob: Exploitation probability (0 = explore, 1 = exploit).
        - alpha: Pheromone influence exponent.
        - beta: Desirability influence exponent.
        - phi: Savings influence exponent for desirability.
        - chi: Distance influence exponent for desirability.
        - gamma: Parcel influence factor for desirability.
        - kappa: Parcel influence exponent for desirability.
        - sigma: Number of elitists for pheromone update.
        - rho: Evaporation rate for pheromone update.
        - value_function: Function to evaluate potential of partial solutions.
        - finalize_policy: Function to complete partial solutions.

        - seed: Random seed for reproducibility.
        - time_limit: Time limit in seconds.
        - verbose: If True, print detailed logs.

    Return:
        - tracker: The SwarmTracker instance with best partials and finalized solutions.
        - info: Dictionary with run statistics.
    """
    start = time.time()
    if depth is None:
        depth = problem.num_actions


    # //// Run an initial greedy solver to estimate initial cost for pheromone initialization
    if verbose:
        print("[ACO] [Init] Estimating costs from initial greedy solver...")
    init_sol, _info = iterative_greedy_solver(
        problem=problem,
        iterations=1000,
        time_limit=2.5,
        seed=10*seed if seed else None,
        verbose=False,
    )
    assert init_sol


    # //// Relocate operator refinements
    if verbose:
        print(f"[MCTS] Applying relocate operator to initial solution...")
    init_partial = PartialSolution.from_solution(init_sol)
    refined_partial, _, _ = relocate_operator(
        init_partial,
        mode='first',
        seed=None if seed is None else 4 * seed + 123
    )
    init_sol = refined_partial.to_solution(); assert init_sol
    init_cost = init_sol.max_cost
    if verbose:
        print(
            f"[MCTS] After relocate, final solution cost: {init_cost}"
        )
    if verbose:
        print(f"[ACO] [Init] Initial solution cost: {init_cost:.3f}")
    assert init_sol, "[ACO] [Init] Initial solver failed to find a solution."
    

    # //// Initialize caches for nearest expansion
    if verbose:
        print("[ACO] [Init] Initializing nearest expansion cache...")
    cache = NearestExpansionCache(problem, n_nearest=5)


    # //// Initialize pheromone and desirability matrices
    if verbose:
        print("[ACO] [Init] Initializing matrices...")
    tau = PheromoneMatrix(problem, sigma=sigma, rho=rho, init_cost=init_cost)
    eta = DesirabilityMatrix(problem, phi, chi, gamma, kappa)


    # //// Initialize solution tracker with the initial greedy solution
    if verbose:
        print("[ACO] [Init] Initializing trackers...")
    lfunc = SolutionTracker()
    lfunc.update(init_sol)
    tracker = SwarmTracker(
        swarm=swarm,
        value_function=value_function,
        finalize_policy=finalize_policy,
    )


    # //// Main ACO iterations
    iterations_completed = 0
    status = "done"
    for run in range(iterations):
        if time.time() - start >= 0.75 * time_limit:
            status = "overtime"
            if verbose:
                print(f"[ACO] Time limit approaching, stopping at run {run}/{iterations}.")
            break

        if verbose:
            print(f"[ACO] [Run {run + 1}/{iterations}] Starting the population run...")

        # Initialize a new population with all components
        population = AntPopulation(
            swarm=swarm,
            cache=cache,
            tau=tau,
            eta=eta,
            lfunc=lfunc,
            alpha=alpha,
            beta=beta,
            omega=omega,
            q_prob=q_prob,
            width=width,
            depth=depth,
            time_limit=time_limit,
            seed=hash(seed + 10 * run) if seed else None,
            verbose=verbose,
        )

        # Run the ACO process and get updated swarm
        if verbose:
            print(f"[ACO] [Run {run + 1}/{iterations}] Running the ant population")
        result_swarm = population.run()

        # Update the trackers with the new swarm
        if verbose:
            print(f"[ACO] [Run {run + 1}/{iterations}] Updating swarm tracker")
            print()

        tracker.update(result_swarm)
        lfunc.update(result_swarm)

        iterations_completed = run + 1


    # //// Finalization

    # Finalize the best partials into complete solutions
    if verbose:
        print(f"[ACO] Finalizing top {n_cutoff} partial into solutions...")
        print()
    tracker.finalize(n_cutoff)

    # Inject the best solution found by ants (if better)
    if lfunc.best_solution:
        tracker.finals.append(lfunc.best_solution)
        tracker.finals.sort(key=lambda s: s.max_cost)
        if n_cutoff and len(tracker.finals) > n_cutoff:
             tracker.finals = tracker.finals[:n_cutoff]

    # Summary info
    elapsed = time.time() - start
    best_sol = tracker.opt(cutoff=n_cutoff)
    init_cost = best_sol.max_cost if best_sol else float("inf")
    stats: Dict[str, Any] = {
        "iterations": iterations_completed,
        "best_cost": init_cost,
        "elitists_count": tracker.num_partials,
        "time": elapsed,
        "status": status,
    }

    # Logging
    if verbose:
        print(
            f"[ACO] The run finished. "
            f"Iterations={stats['iterations']}, "
            f"Best_cost={stats['best_cost']:.3f}, "
            f"Time={stats['time']:.3f}s."
        )

    return tracker, stats




# ================ ACO API Functions ================
def aco_enumerator(
    problem: ShareARideProblem,
    swarm: Optional[PartialSolutionSwarm] = None,

    # Run parameters
    n_partials: int = 50,
    n_cutoff: int = 10,
    n_return: int = 5,
    iterations: int = 10,
    depth: Optional[int] = None,

    # Tuning hyperparameters
    q_prob: float = 0.71,
    alpha: float = 1.36,
    beta: float = 1.38,
    omega: float = 3,
    phi: float = 0.43,
    chi: float = 1.77,
    gamma: float = 0.40,
    kappa: float = 2.34,
    sigma: int = 12,
    rho: float = 0.62,
    width: int = 4,

    # Policies
    value_function: ValueFunction = _default_value_function,
    finalize_policy: FinalizePolicy = _default_finalize_policy,

    # Meta parameters
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[List[Solution], Dict[str, Any]]:
    """
    Run ACO to enumerate the best k complete solutions.
    
    This function executes ACO with a SwarmTracker that tracks the best
    variation of each partial across all depth. At the end, it finalizes
    the top `n_cutoff` partials and returns the best `n_return` solutions.
    
    Args:
        - problem: ShareARideProblem instance.
        - swarm: Initial PartialSolutionSwarm (optional, created if None).

        - n_partials: Number of ants (must match swarm size if provided).
        - n_return: Number of best solutions to return.
        - n_cutoff: Maximum partials to finalize (time-saving cutoff). Should be
                >= n_return to avoid missing potentially better solutions.
        - iterations: Number of ACO iterations to run.
        - depth: Number of depth to run.
        
        - q_prob: Exploitation probability (0 = explore, 1 = exploit).
        - alpha: Pheromone influence exponent.
        - beta: Desirability influence exponent.
        - phi: Savings influence exponent for desirability.
        - chi: Distance influence exponent for desirability.
        - gamma: Parcel influence factor for desirability.
        - kappa: Parcel influence exponent for desirability.
        - sigma: Number of elitists for pheromone update.
        - rho: Evaporation rate for pheromone update.
        - width: Maximum actions to consider per expansion.

        - value_function: Function to evaluate potential of partial solutions.
        - finalize_policy: Function to complete partial solutions.

        - seed: Random seed for reproducibility.
        - time_limit: Total time limit in seconds.
        - verbose: If True, print detailed logs.
    
    Returns:
        - solutions: List of top n_return solutions from finalized partials.
        - info: Dictionary with run statistics.
    """
    # Create initial swarm if not provided
    if swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(n_partials)
        ]
        swarm = PartialSolutionSwarm(solutions=initial_partials)
    else:
        # Assert swarm size matches num_ants for transparency
        assert swarm.num_partials == n_partials, (
            f"Swarm size ({swarm.num_partials}) must match num_ants ({n_partials})"
        )


    # //// Run ACO
    tracker, run_info = _run_aco(
        problem=problem,
        swarm=swarm,

        n_cutoff=n_cutoff,
        iterations=iterations,
        depth=depth,

        q_prob=q_prob,
        alpha=alpha,
        beta=beta,
        omega=omega,
        phi=phi,
        chi=chi,
        gamma=gamma,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        width=width,

        value_function=value_function,
        finalize_policy=finalize_policy,

        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )

    # Get top n_return solutions
    top_solutions = tracker.top(
        k=n_return,
        cutoff=n_cutoff,
    )

    # Build info from run_info
    stats: Dict[str, Any] = {
        "iterations": run_info["iterations"],
        "time": run_info['time'],
        "best_cost": run_info["best_cost"],
        "solutions_found": len(top_solutions),
        "elitists_count": run_info["elitists_count"],
        "status": run_info['status'],
    }

    # Logging
    if verbose:
        print()
        print("[ACO] Enumeration complete.")
        print(f"[ACO] Total solutions found: {stats['solutions_found']}.")
        print(
            f"[ACO] Solution costs range: "
            f"{top_solutions[0].max_cost:.3f} "
            f"- {top_solutions[-1].max_cost:.3f}."
        )
        print(f"[ACO] Total time: {stats['time']:.3f}s")
        print("------------------------------")
        print()

    return top_solutions, stats


def aco_solver(
    problem: ShareARideProblem,
    swarm: Optional[PartialSolutionSwarm] = None,

    # Run parameters
    n_partials: int = 50,
    n_cutoff: int = 10,
    iterations: int = 40,
    depth: Optional[int] = None,

    # Hyperparameters
    q_prob: float = 0.72,
    alpha: float = 1.36,
    beta: float = 1.38,
    omega: float = 3,
    phi: float = 0.43,
    chi: float = 1.77,
    gamma: float = 0.40,
    kappa: float = 2.34,
    sigma: int = 12,
    rho: float = 0.62,
    width: int = 4,

    # Policies
    value_function: ValueFunction = _default_value_function,
    finalize_policy: FinalizePolicy = _default_finalize_policy,

    # Meta parameters
    seed: Optional[int] = None,
    time_limit: float = 30.0,
    verbose: bool = False,
) -> Tuple[Optional[Solution], Dict[str, Any]]:
    """
    Run ACO solver and return the best complete solution.
    
    This function executes the ACO algorithm using SwarmTracker to track
    the best partials across depth and returns the optimal solution.
    
    Args:
        - problem: ShareARideProblem instance.
        - swarm: Initial PartialSolutionSwarm (optional, created if None).

        - n_partials: Number of ants (must match swarm size if provided).
        - n_cutoff: Maximum partials to finalize (time-saving cutoff). Should be
                > 1 to avoid missing potentially better solutions.
        - iterations: Number of ACO iterations to run.
        - depth: Number of depth to run.
        
        - q_prob: Exploitation probability (0 = explore, 1 = exploit).
        - alpha: Pheromone influence exponent.
        - beta: Desirability influence exponent.
        - phi: Savings influence exponent for desirability.
        - chi: Distance influence exponent for desirability.
        - gamma: Parcel influence factor for desirability.
        - kappa: Parcel influence exponent for desirability.
        - sigma: Number of elitists for pheromone update.
        - rho: Evaporation rate for pheromone update.
        - width: Maximum actions to consider per expansion.

        - value_function: Function to evaluate potential of partial solutions.
        - finalize_policy: Function to complete partial solutions.

        - seed: Random seed for reproducibility.
        - time_limit: Total time limit in seconds.
        - verbose: If True, print detailed logs.
    
    Returns:
        - solution: Best solution found from the tracker (or None if failed).
        - info: Dictionary with run statistics.
    """
    # Create initial swarm if not provided
    if swarm is None:
        initial_partials = [
            PartialSolution(problem=problem, routes=[]) for _ in range(n_partials)
        ]
        swarm = PartialSolutionSwarm(solutions=initial_partials)
    else:
        # Assert swarm size matches num_ants for transparency
        assert swarm.num_partials == n_partials, (
            f"Swarm size ({swarm.num_partials}) must match num_ants ({n_partials})"
        )


    # //// Run ACO
    tracker, run_info = _run_aco(
        problem=problem,
        swarm=swarm,

        n_cutoff=n_cutoff,
        iterations=iterations,
        depth=depth,

        q_prob=q_prob,
        alpha=alpha,
        beta=beta,
        omega=omega,
        phi=phi,
        chi=chi,
        gamma=gamma,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        width=width,

        value_function=value_function,
        finalize_policy=finalize_policy,

        seed=seed,
        time_limit=time_limit,
        verbose=verbose,
    )

    # Get optimal solution from tracker
    best_solution = tracker.opt()

    # Build info from run_info
    stats: Dict[str, Any] = {
        "iterations": run_info["iterations"],
        "time": run_info['time'],
        "best_cost": run_info["best_cost"],
        "elitists_count": run_info["elitists_count"],
        "status": run_info['status']
    }

    # Logging
    if verbose:
        print()
        print("[ACO] Solver complete.")
        if best_solution is not None:
            print(f"[ACO] Best solution cost: {best_solution.max_cost}")
        else:
            print("[ACO] No valid solution found.")
        print(f"[ACO] Total time: {stats['time']:.3f}s")
        print("------------------------------")
        print()

    return best_solution, stats




# ================ Playground ================
if __name__ == "__main__":
    from share_a_ride.solvers.algo.utils import test_problem

    _, _ = aco_solver(
        problem=test_problem,
        time_limit=60.0,
        seed=120,
        verbose=True,
    )
