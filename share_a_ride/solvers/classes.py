"""
Module defining solver-related classes and enums.
These classes and enums encapsulate data related to solver types, tags, modes, hyperparameters,
and solver information. This can be seen as a schema/configuration module for solvers.
"""
import importlib
from typing import Dict, Tuple, Any, Optional, Callable, Union, List
from enum import Enum
from dataclasses import dataclass, field

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution




# ================ Solver schema =================
class SolverName(Enum):
    """
    Enum for different solver names.
    """
    GREEDY = "iterative_greedy"
    BEAM = "beam"
    ASTAR = "astar"
    MCTS = "mcts"
    ACO = "aco"
    ALNS = "alns"
    HGS = "hgs"
    BNB = "bnb"
    EXHAUST = "exhaust"
    MILP = "milp"
    CIP = "cip"

    @ staticmethod
    def solvers() -> List['SolverName']:
        """
        Get a list of all solver names.
        """
        return [solver for solver in SolverName]




class AlgoTag(Enum):
    """
    Enum for different solver types.
    It is a high-level categorization system for solvers, serving for
    organizational and filtering purposes.

    The tags include:
    - DEFENSIVE: Solvers that has a defense mechanism (using `defense_policy`)
    - EXPANSIVE: Solvers that search without backtracking (no specific parameter yet)
    - ITERATIVE: Solvers that iteratively improve the solution (using `iterations`).
    - NEARBASED: Solvers that constrain expansion in nearest neighbors (using `width`)
    - POPULATION: Solvers that maintain a population (using `swarm` instead of `partial`).
    - PRUNE: Solvers that prune the search space to focus on promising areas (using `incumbent`).
    """
    DEFENSIVE = "defensive"
    EXPANSIVE = "expansive"
    ITERATIVE = "iterative"
    NEARBASED = "nearbased"
    POPULATION = "population"
    PRUNE = "prune"
    EXACT = "exact"




class SolverTags(Enum):
    """
    Enum for different solver tags.
    This serves as a mapping from solver names to their associated tags.
    """
    ALNS = []
    ACO = [AlgoTag.ITERATIVE, AlgoTag.POPULATION, AlgoTag.DEFENSIVE, AlgoTag.NEARBASED]
    ASTAR = [AlgoTag.EXPANSIVE, AlgoTag.DEFENSIVE, AlgoTag.NEARBASED]
    BEAM = [AlgoTag.DEFENSIVE, AlgoTag.EXPANSIVE, AlgoTag.POPULATION, AlgoTag.NEARBASED]
    BNB = [AlgoTag.PRUNE, AlgoTag.EXACT]
    EXHAUST = [AlgoTag.PRUNE, AlgoTag.EXACT]
    GREEDY = [AlgoTag.ITERATIVE, AlgoTag.EXPANSIVE]
    HGS = [AlgoTag.POPULATION]
    MCTS = [AlgoTag.EXPANSIVE, AlgoTag.DEFENSIVE, AlgoTag.NEARBASED]
    MILP = [AlgoTag.EXACT]
    CIP = [AlgoTag.EXACT]



class SolverFunc(Enum):
    """
    Enum for different solver functions.
    This serves as a mapping from solver names to their corresponding function implementations.
    """
    GREEDY = ("share_a_ride.solvers.algo.greedy", "iterative_greedy_solver")
    BEAM = ("share_a_ride.solvers.algo.beam", "beam_solver")
    ASTAR = ("share_a_ride.solvers.algo.astar", "astar_solver")
    MCTS = ("share_a_ride.solvers.algo.mcts", "mcts_solver")
    ACO = ("share_a_ride.solvers.algo.aco", "aco_solver")
    ALNS = ("share_a_ride.solvers.algo.alns", "alns_solver")
    HGS = ("share_a_ride.solvers.algo.hgs", "hgs_solver")
    BNB = ("share_a_ride.solvers.algo.bnb", "bnb_solver")
    EXHAUST = ("share_a_ride.solvers.algo.exhaust", "exhaust_solver")
    MILP = ("share_a_ride.solvers.algo.milp", "milp_solver")
    CIP = ("share_a_ride.solvers.algo.cip", "cip_solver")




class SolverMode(Enum):
    """
    Enum for different solver modes.
    """
    LIGHT = 0.25
    STANDARD = 1.0
    HEAVY = 4.0
    INTENSIVE = 16.0




class SolverParams(Enum):
    """
    Enum for different solver hyperparameter presets.
    """


    # //// Define hyperparameter presets for each solver ////
    GREEDY = {
        "scaling": {"iterations": 10000, "time_limit": 60.0},
        "hyperparameters": {
            "num_actions": 7,
            "t_actions": 0.01,
            "destroy_proba": 0.53,
            "destroy_steps": 13,
            "destroy_t": 1.3,
            "rebuild_proba": 0.29,
            "rebuild_steps": 3,
            "rebuild_t": 1.2,
        },
    }
    BEAM = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {
            "n_partials": 100,
            "width": 5,
            "r_intra": 0.55,
            "r_inter": 0.75,
            "f_intra": 0.05,
            "f_inter": 0.15,
        },
    }
    ASTAR = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {
            "width": 5,
            "eps": 0.285,
            "cutoff_depth": 10,
            "cutoff_size": 3000,
            "cutoff_ratio": 0.285,
        },
    }
    MCTS = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {
            "width": 3,
            "uct_c": 0.24,
            "cutoff_depth": 9,
            "cutoff_depth_inc": 3,
            "cutoff_iter": 900,
            "reward_pow": 1.69,
        },
    }
    ACO = {
        "scaling": {"iterations": 10, "time_limit": 60.0},
        "hyperparameters": {
            "width": 5,
            "q_prob": 0.54,
            "alpha": 1.14,
            "beta": 1.83,
            "omega": 4,
            "phi": 0.36,
            "chi": 1.65,
            "gamma": 0.43,
            "kappa": 2.03,
            "sigma": 11,
            "rho": 0.81,
        },
    }
    ALNS = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {
            "decay": 0.8,
        },
    }
    HGS = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {
            "n_partials": 40,
        },
    }
    BNB = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {},
    }
    EXHAUST = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {},
    }
    MILP = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {},
    }
    CIP = {
        "scaling": {"time_limit": 60.0},
        "hyperparameters": {},
    }


    # ================ Hyperparameter utilities ================
    @ staticmethod
    def hyperparams(name: SolverName) -> Dict[str, Any]:
        """
        Get the hyperparameters for the given solver name.
        """
        return SolverParams[name.name].value["hyperparameters"]


    # ================ Point scaling =================
    @staticmethod
    def scale(name: SolverName, mode: SolverMode) -> Dict[str, Any]:
        """
        Scale hyperparameters based on solver mode and tags.

        Here we only scale `time_limit` and `iterations` (if AlgoTag.ITERATIVE is present).
        """
        base = SolverParams[name.name].value
        scaled = base["scaling"].copy()

        # Scale relevant parameters
        scaled["time_limit"] *= mode.value

        tags = SolverTags[name.name].value
        if AlgoTag.ITERATIVE in tags:
            scaled["iterations"] = int(scaled["iterations"] * mode.value)

        return scaled


    @staticmethod
    def run_params(
            name: SolverName,
            mode: SolverMode,
        ) -> Dict[str, Any]:
        """
        Prepare parameters for running the solver defined by name in the given mode.
        Flattens the parameter groups into a single dictionary.
        """
        base = SolverParams[name.name].value
        scaled = SolverParams.scale(name, mode)

        final = {
            **scaled,
            **base["hyperparameters"]
        }
        return final




    # ================ Interval scaling ================
    @staticmethod
    def interval(
            val: Union[int, float], c: float = 1.33
        ) -> Union[Tuple[int, int], Tuple[float, float]]:
        """
        Compute hyperparameters interval for tuning. Scale the base hyperparameter
        defined above by from 1 / c to c, while keeping integer parameters as integers.
        """
        if isinstance(val, int):
            low = int(val / c)
            high = int(val * c + 0.5)
            if low == high:
                high += 1
            return (low, high)

        # For floats
        return (val / c, val * c)


    @staticmethod
    def hyperparameter_intervals(
            name: SolverName,
        ) -> Dict[str, Union[Tuple[int, int], Tuple[float, float]]]:
        """
        Compute hyperparameters intervals for tuning.
        """
        base = SolverParams[name.name].value
        ranges = {}

        for key, val in base["hyperparameters"].items():
            itvl = SolverParams.interval(val)

            # Clamp probabilities
            if "prob" in key or "decay" in key:
                itvl = (max(0.0, itvl[0]), min(1.0, itvl[1]))
            ranges[key] = itvl

        return ranges


    @staticmethod
    def tune_params(
            name: SolverName,
            mode: SolverMode,
        ) -> Dict[str, Any]:
        """
        Compute hyperparameters interval for tuning.
        """
        scaled = SolverParams.scale(name, mode)
        intervals = SolverParams.hyperparameter_intervals(name)

        final = {
            **scaled,
            **intervals
        }

        return final




@dataclass
class SolverInfo:
    """
    Data class to hold solver information and configuration.
    """
    name: SolverName
    mode: SolverMode = SolverMode.STANDARD
    func: Callable = field(init=False)
    tags: list = field(init=False)
    params: dict = field(init=False)


    def __post_init__(self):
        module_path, func_name = SolverFunc[self.name.name].value
        module = importlib.import_module(module_path)
        self.func = getattr(module, func_name)
        self.tags = SolverTags[self.name.name].value
        self.params = SolverParams.run_params(self.name, self.mode)


    def with_mode(self, name: SolverName, mode: SolverMode) -> 'SolverInfo':
        """
        Return a new SolverInfo with parameters scaled for the given mode.
        """
        return SolverInfo(name, mode)




class Solver:
    """
    Wrapper for all solvers. Provides lookup, parameter scaling, and unified solve interface.
    """
    def __init__(self, name: SolverName, mode: SolverMode = SolverMode.STANDARD):
        self.info = SolverInfo(name, mode)


    def run(
            self,
            problem: ShareARideProblem,
            seed: Optional[int] = None,
            verbose: bool = False,
            **kwargs,
        ) -> Tuple[Optional[Solution], Dict[str, Any]]:
        """
        Solve the given problem using the specified solver name and mode.
        Additional keyword arguments can be provided to override default parameters.
        """


        # Get parameters from SolverParams
        params = SolverParams.run_params(self.info.name, self.info.mode)

        # Override with provided arguments
        if seed is not None:
            params["seed"] = seed
        params["verbose"] = verbose
        params.update(kwargs)

        # Solve the problem
        return self.info.func(problem, **params)
