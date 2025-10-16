"""
Main script to demonstrate/test the SARP solvers and utilities in each development sprint.
"""
from share_a_ride.data.executor import attempt_dataset
from share_a_ride.data.summarizer import summarize_dataset

from share_a_ride.utils.generator import generate_instance_coords

from share_a_ride.solvers.algo.Algo import AlgoSolver
from share_a_ride.solvers.algo.exhaustive import exhaustive_solver
from share_a_ride.solvers.algo.bnb import branch_and_bound_solver
from share_a_ride.solvers.algo.greedy import greedy_balanced_solver, iterative_greedy_balanced_solver




def demo_sprint_1():
    # Prepare type I test instances
    type_I = []
    params_I = [(2,2,1), (2,3,2), (3,2,2), (2,3,2), (3,3,2)]
    for idx, (N, M, K) in enumerate(params_I):
        prob = generate_instance_coords(N, M, K, area=100, seed=100+idx)
        # prob = generate_instance_lazy(
        #     N, M, K, low=10, high=99, lmbd=20.0,
        #     use_poisson=True, seed=100+idx)
        type_I.append(prob)

    # Prepare type II test instances
    type_II = []
    params_II = [(3,4,2), (4,3,2), (2,6,2), (3,5,3), (4,5,3)]
    for idx, (N, M, K) in enumerate(params_II): 
        prob = generate_instance_coords(N, M, K, area=100, seed=200+idx)
        # prob = generate_instance_lazy(
        #     N, M, K, low=10, high=99, lmbd=20.0,
        #     use_poisson=True, seed=200+idx
        # )
        type_II.append(prob)


    # Solve Type I
    print("Solving Type I tests with brute-force enumerator:\n")
    type_I_results = []
    for prob in type_I:
        # Describe
        prob.pretty_print(verbose=1)

        # Solve
        sol, info = exhaustive_solver(prob, max_solutions=2000000,
                time_limit=10.0, verbose=False)

        # Show result info
        print("Enumeration info:", info)
        if sol:
            sol.stdin_print(verbose=1)
        else:
            print("No solution enumerated within time limit.")
        print("\n")

    # Solve Type II
    print("Solving Type II tests with branch and bound\n")
    type_II_results = []
    for prob in type_II:
        # Describe
        prob.pretty_print(verbose=1)

        # Solve
        sol, info = branch_and_bound_solver(prob, time_limit=10.0, verbose=False)

        # Show result info
        print("Enumeration info:", info)

        # Show results
        if sol:
            sol.stdin_print(verbose=1)
        else:
            print("No solution found within time limit.")
        print("\n")

        type_II_results.append((sol, info))



def demo_sprint_2():
    # AlgoSolver container demo
    chosen_solver = AlgoSolver(
        algo=iterative_greedy_balanced_solver,
        args={"iterations": 10000, "time_limit": 10.0, "seed": 42, "verbose": 1},
        hyperparams={
            "destroy_proba"      : 0.4,
            "destroy_steps"     : 3,
            "destroy_T"         : 1.0,
            "rebuild_proba"      : 0.4,
            "rebuild_steps"     : 1,
            "rebuild_T"         : 5.0,
        }
    )

    # Dataset attempt demo
    sols, gaps, msg = attempt_dataset(chosen_solver, "H", note="test attempt", verbose=True)

    # Visualize functionality demo
    for sol in sols:
        if sol:
            sol.stdin_print(verbose=1)
            sol.visualize()


def demo_sprint_3():
    solver_bnb = AlgoSolver(
        algo=branch_and_bound_solver,
        args={"time_limit": 5.0, "verbose": 0},
        hyperparams={}
    )
    sols1, gaps1, msg1 = attempt_dataset(
        solver_bnb,
        "H",
        note="test BnB and Iterative greedy on H dataset",
        verbose=True
    )
    summarize_dataset("H", verbose=True)

    solver_iter_greedy = AlgoSolver(
        algo=iterative_greedy_balanced_solver,
        args={"iterations": 10000, "time_limit": 10.0, "seed": 42, "verbose": 1},
        hyperparams={
            "destroy_proba"      : 0.5,
            "destroy_steps"     : 5,
            "destroy_T"         : 1.0,
            "rebuild_proba"      : 0.3,
            "rebuild_steps"     : 2,
            "rebuild_T"         : 5.0,
        }
    )
    sols2, gaps2, msg2 = attempt_dataset(
        solver_iter_greedy,
        "H",
        note="test BnB and Iterative greedy on H dataset",
        verbose=True
    )
    summarize_dataset("H", verbose=True)



if __name__ == "__main__":
    demo_sprint_3()
