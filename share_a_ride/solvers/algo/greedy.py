import time, random
from typing import List

from share_a_ride.problem import ShareARideProblem
from share_a_ride.solution import Solution
from share_a_ride.solvers.operator.build import build
from share_a_ride.solvers.operator.destroy import destroy
from share_a_ride.utils.helper import route_cost_from_sequence

def greedy_balanced_solver(
        prob: ShareARideProblem, 
        premature_routes: List[List[int]] = None,   # List of K premature routes, starting at depot 0
        verbose: bool = False
    ) -> tuple:
    """
    Greedy balanced heuristic:
        - At each step, choose the taxi with smallest current route cost
        - For that taxi, evaluate all feasible next actions (pick/drop).
        - Choose the action with minimal added distance (inc).
        - Repeat until all passengers and parcels are served.

    Returns:
        (sol, info): tuple where
        - sol: Solution object with routes and costs.
        - Info dictionary contains:
            + iterations: number of main loop iterations
            + actions_evaluated: total number of actions evaluated
            + elapsed_time: total time taken
    """
    start_time = time.time()
    N, M, K = prob.N, prob.M, prob.K

    # state
    if premature_routes:
        # Initialize remaining_pass_pick
        remaining_pass_pick = set(
            e for e in range(1, N + 1)
            if all(prob.ppick(e) not in route for route in premature_routes)
        )

        # Initialize remaining_parc_pick
        remaining_parc_pick = set(
            e for e in range(1, M + 1)
            if all(prob.parc_pick(e) not in route for route in premature_routes)
        )

        # Initialize taxi_current_pass
        taxi_current_pass = []
        for route in premature_routes:
            last_pick = 0
            for node in route:
                for e in range(1, N + 1):
                    if node == prob.ppick(e):
                        last_pick = e
                    elif node == prob.pdrop(e) and last_pick == e:
                        last_pick = 0

            taxi_current_pass.append(last_pick)

        # Initialize taxi_current_parc
        taxi_current_parc = []
        for route in premature_routes:
            onboard = set()
            for node in route:
                for e in range(1, M + 1):
                    if node == prob.parc_pick(e):
                        onboard.add(e)
                    elif node == prob.parc_drop(e):
                        onboard.discard(e)

            taxi_current_parc.append(onboard)

        # Initialize taxi_states
        taxi_states = []
        for k, route in enumerate(premature_routes):
            pos = route[-1] if route else 0
            cost = route_cost_from_sequence(route, prob.D)
            load = sum(prob.q[j - 1] for j in taxi_current_parc[k]) \
                if taxi_current_parc[k] else 0
            passenger = taxi_current_pass[k]
            ended = (route[-1] == 0) if route and len(route) > 1 else False

            taxi_states.append({
                "route": route,
                "pos": pos,
                "cost": cost,
                "load": load,
                "passenger": passenger,
                "ended": ended
            })

    else:
        remaining_pass_pick = set(range(1, N + 1))
        remaining_parc_pick = set(range(1, M + 1))
        taxi_current_pass = [0] * K
        taxi_current_parc = [set() for _ in range(K)]
        taxi_states = []
        for k in range(K):
            taxi_states.append({
                "route": [0],
                "pos": 0,
                "cost": 0,
                "load": 0,
                "passenger": 0,
                "ended": False
            })

    # Statistics tracking
    stats = {
        "iterations": 0,
        "actions_evaluated": 0,
    }


    def possible_actions(t_state: dict, t_idx: int):
        """
            Generate feasible next actions for a taxi t.
            Returns a list of (action_type, node_index, incremental_cost).
        """
        pos, actions = t_state["pos"], []

        # passenger pickup
        for i in list(remaining_pass_pick):
            if t_state["passenger"] == 0:  # no passenger onboard
                inc = prob.D[pos][prob.ppick(i)]
                actions.append(("pickP", i, inc))

        # passenger drop
        if t_state["passenger"] > 0:  # passenger onboard
            inc = prob.D[pos][prob.pdrop(t_state["passenger"])]
            actions.append(("dropP", t_state["passenger"], inc))

        # parcel pickup
        for j in list(remaining_parc_pick):
            qj = prob.q[j - 1]
            if t_state["load"] + qj <= prob.Q[t_idx]:
                inc = prob.D[pos][prob.parc_pick(j)]
                actions.append(("pickL", j, inc))

        # parcel drop
        for j in taxi_current_parc[t_idx]:
            inc = prob.D[pos][prob.parc_drop(j)]
            actions.append(("dropL", j, inc))

        return actions


    def apply_action(t_state: dict, t_idx: int, kind: str, node_idx: int, inc: int):
        """Apply an action to taxi t and update global sets."""
        nonlocal remaining_pass_pick, remaining_parc_pick
        nonlocal taxi_current_pass, taxi_current_parc

        if kind == "pickP":
            t_state["route"].append(prob.ppick(node_idx))
            t_state["passenger"] = node_idx
            remaining_pass_pick.remove(node_idx)
            taxi_current_pass[t_idx] = node_idx

        elif kind == "dropP":
            t_state["route"].append(prob.pdrop(node_idx))
            t_state["passenger"] = 0
            taxi_current_pass[t_idx] = 0

        elif kind == "pickL":
            t_state["route"].append(prob.parc_pick(node_idx))
            t_state["load"] += prob.q[node_idx - 1]
            remaining_parc_pick.remove(node_idx)
            taxi_current_parc[t_idx].add(node_idx)

        elif kind == "dropL":
            t_state["route"].append(prob.parc_drop(node_idx))
            t_state["load"] -= prob.q[node_idx - 1]
            taxi_current_parc[t_idx].remove(node_idx)

        else:
            raise ValueError(f"Unknown action kind: {kind}")

        t_state["cost"] += inc
        t_state["pos"] = t_state["route"][-1]


    # Main loop: execute actions until all pickups and drops are resolved.
    while ( remaining_pass_pick or any(taxi_current_pass) \
            or remaining_parc_pick or any(taxi_current_parc)):

        stats["iterations"] += 1

        # List all available taxis
        available_taxis = [
            t_idx for t_idx, t_state in enumerate(taxi_states)
            if not t_state["ended"]
        ]

        # If no taxis are available, break.
        if not available_taxis:
            break

        # Select taxi with minimal route cost.
        argmin_t_idx = min(available_taxis, key=lambda t_idx: taxi_states[t_idx]["cost"])
        argmin_t_state = taxi_states[argmin_t_idx]

        # Get feasible actions and record how many were evaluated.
        actions = possible_actions(argmin_t_state, argmin_t_idx)
        stats["actions_evaluated"] += len(actions)

        if verbose:
            print(f"Taxi with min cost: {argmin_t_idx}")
            print(f"Actions available: {actions}")

        # If no feasible action exists, send taxi back to depot.
        if not actions:
            assert argmin_t_state["passenger"] == 0
            assert argmin_t_state["load"] == 0

            argmin_t_state["route"].append(0)
            argmin_t_state["pos"] = 0
            argmin_t_state["cost"] += prob.D[argmin_t_state["pos"]][0]
            argmin_t_state["ended"] = True

            continue

        # Select action with minimal incremental cost.
        kind, idx, inc = min(actions, key=lambda x: x[2])
        apply_action(argmin_t_state, argmin_t_idx, kind, idx, inc)
        
        if verbose:
            print(f"Taxi: {argmin_t_idx}: {argmin_t_state['route']}")
            print()


    for t_state in taxi_states:
        if not t_state["ended"]:
            t_state["route"].append(0)
            t_state["ended"] = True

    if verbose:
        print("All tasks completed.")


    # Build final solution.
    routes, route_costs = [], []
    for t_state in taxi_states:
        routes.append(t_state["route"])
        cost = route_cost_from_sequence(t_state["route"], prob.D)
        route_costs.append(cost)

    sol = Solution(prob, routes, route_costs)
    elapsed = time.time() - start_time
    info = {
        "iterations": stats["iterations"],
        "actions_evaluated": stats["actions_evaluated"],
        "elapsed_time": elapsed
    }
    if sol:
        if not sol.is_valid():
            sol = None


    assert sol.is_valid() if sol else True
    return sol, info



def iterative_greedy_balanced_solver(
        prob: ShareARideProblem,
        # args
        iterations: int = 10,
        time_limit: float = 10.0,   # overall time limit in seconds
        seed: int = 42,
        verbose: bool = False,
        # hyperparams
        destroy_prob: float = 0.4, # probability of selecting one route for destruction
        destroy_steps: int = 15,    # number of destroyed nodes for the selected route
        destroy_T: float = 1.0,     # temperature for destroy heuristic
        rebuild_prob: float = 0.3,  # probability of rebuilding a destroyed route
        rebuild_steps: int = 5,     # number of random actions to perform during rebuilding
        rebuild_T: float = 1.0,     # temperature for rebuild heuristic
    ) -> tuple:
    '''
    Iterative greedy balanced heuristic:
        - Start with a greedy balanced solution.
        - Destroy a portion of routes of the solution 
            using a temperature-based selection heuristic on route costs.
        - Rebuild a portion of the destroyed routes using random actions.
        - Greedily optimize the new partial solution.
        - Repeat for a number of iterations or until no improvement.
    Operators used:
        - destroy: remove a suffix of a route (keeping depot 0)
        - randbuild: randomly insert pickups/drops up to num_actions steps
    Returns:
        (sol, info): tuple where
        - sol: Solution object with routes and costs.
        - Info dictionary contains:
            + iterations: number of main loop iterations
            + improvements: number of improvements found
            + actions_evaluated: total number of actions evaluated
            + nodes_destroyed: num. of nodes destroyed
            + nodes_rebuilt: num. of nodes rebuilt
            + elapsed_time: total time taken
            + status: "done" or "timeout"
    '''
    assert 0 <= destroy_prob <= 1
    assert 0 <= rebuild_prob <= 1
    assert 1 <= rebuild_steps <= destroy_steps

    # initialize time
    start_time = time.time()
    deadline = start_time + time_limit if time_limit is not None else None

    # initial solution
    best_sol, base_info = greedy_balanced_solver(prob, verbose=False)
    if not best_sol:
        return None, {}
    best_cost = best_sol.max_cost

    # initialize statistics
    total_actions = base_info.get("actions_evaluated", 0)
    improvements = 0
    nodes_destroyed = 0
    nodes_rebuilt = 0
    status = "done"
    iterations_done = 0

    # helper function to destroy routes by cost
    def _destroy_routes_heuristic(
            prob: ShareARideProblem,
            sol: Solution,
            destroy_prob: float,
            destroy_steps: int,
            seed_: int = 42,
            T: float = 1.0
        ) -> tuple[List[List[int]], List[bool], int]:
        '''
        Copy routes from sol and remove suffixes using temperature-based selection.
        
        Temperature controls the selection bias:
        - temperature = 0: always select the most expensive routes (greedy)
        - temperature → ∞: uniform random selection
        - temperature = 1: balanced probabilistic selection
        
        Higher cost routes have higher probability of being selected, but with
        some randomness to allow exploration.
        
        Args:
            prob: ShareARideProblem instance
            sol: Current solution
            destroy_prob: Fraction of routes to destroy (0 to 1)
            destroy_steps: Maximum number of nodes to remove per route
            seed: for reproducibility
            temperature: Controls selection randomness (default 1.0)
        
        Returns:
            (routes, flags, num_removed): 
                - routes: List of routes with some destroyed
                - flags: Boolean list indicating which routes were destroyed
                - num_removed: Total number of nodes removed
        '''
        rng_ = random.Random(seed_)
        
        # Copy routes and initialize flags
        routes = [route[:] for route in sol.routes]
        flags = [False] * len(routes)
        num_removed = 0

        # Determine how many routes to destroy
        count = max(1, int(round(destroy_prob * len(routes) + 0.5)))

        # Temperature-based selection
        costs = [sol.route_costs[idx] for idx in range(len(routes))]


        # Probabilistic selection based on cost and temperature
        # Use softmax-like weighting: weight_i = cost_i^(1/T)
        # Higher costs and lower temperature → higher probability

        # Normalize costs to avoid numerical issues
        min_cost = min(costs) if costs else 0
        max_cost = max(costs) if costs else 1
        cost_range = max_cost - min_cost

        if cost_range < 1e-6:
            # All costs are equal, use uniform selection
            selected_indices = rng_.sample(range(len(routes)), count)
        else:
            # Compute weights with temperature
            weights = []
            for cost in costs:
                # Normalize to [0, 1] range to avoid overflow
                normalized = (cost - min_cost) / cost_range
                # Apply temperature: higher cost → higher weight
                weight = (normalized + 0.1) ** (1.0 / T)
                weights.append(weight)

            # Sample without replacement using weights
            selected_indices = []
            available_indices = list(range(len(routes)))
            available_weights = weights[:]

            for _ in range(count):
                # Weighted random choice
                total = sum(available_weights)
                rand_val = rng_.random() * total
                cumsum = 0

                for i, (idx, w) in enumerate(zip(available_indices, available_weights)):
                    cumsum += w
                    if rand_val <= cumsum:
                        selected_indices.append(idx)
                        available_indices.pop(i)
                        available_weights.pop(i)
                        break

        # Destroy the selected routes
        for idx in selected_indices:
            route = routes[idx]
            if len(route) <= 2:  # Skip routes with only depot + 1 node
                continue

            reduced = destroy(prob, route, max_remove=destroy_steps, verbose=False)
            removed = max(0, len(route) - len(reduced))

            if removed > 0:
                routes[idx] = reduced
                flags[idx] = True
                num_removed += removed

        return routes, flags, num_removed


    # Main iterative loop
    rng = random.Random(seed)
    if verbose:
        print(f"[iter 0] initial best cost: {best_cost}")
    for it in range(1, iterations + 1):
        if deadline and time.time() >= deadline:
            status = "timeout"
            break

        iterations_done = it

        # destroy
        candidate_routes, destroyed_flags, removed = _destroy_routes_heuristic(
            prob, best_sol, destroy_prob, destroy_steps,
            2 * seed + it, T=destroy_T
        )
        nodes_destroyed += removed

        # rebuild
        for idx, was_destroyed in enumerate(destroyed_flags):
            # Choose whether to rebuild this route
            if not was_destroyed or len(candidate_routes[idx]) <= 2:
                continue
            if rng.random() > rebuild_prob:
                continue

            # Perform random rebuilding actions
            old_len = len(candidate_routes[idx])
            candidate_routes[idx] = build(
                prob,
                candidate_routes[idx],
                idx,
                num_actions=rebuild_steps,
                seed=(2 * seed + it + idx) if seed is not None else None,
                T=rebuild_T,
                verbose=False
            )
            nodes_rebuilt += max(0, len(candidate_routes[idx]) - old_len)

        # greedily complete the remaining partial solution
        sol_candidate, info_candidate = greedy_balanced_solver(
            prob,
            premature_routes=candidate_routes,
            verbose=False
        )
        total_actions += info_candidate.get("actions_evaluated", 0)

        # if improved, update best solution
        if sol_candidate and sol_candidate.is_valid() and sol_candidate.max_cost < best_cost:
            best_sol = sol_candidate
            best_cost = sol_candidate.max_cost
            improvements += 1
            if verbose:
                print(f"[iter {it}] improved best to {best_cost}")


    elapsed = time.time() - start_time
    info = {
        "iterations": iterations_done,
        "improvements": improvements,
        "actions_evaluated": total_actions,
        "nodes_destroyed": nodes_destroyed,
        "nodes_rebuilt": nodes_rebuilt,
        "elapsed_time": elapsed,
        "status": status,
    }

    return best_sol, info

