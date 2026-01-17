"""
Container for operator functions in the share_a_ride problem.
Operators are stateless functions that transform routes.
"""

from typing import Callable, Dict, Any, List
from share_a_ride.core.problem import ShareARideProblem

class Operator:
    """
    Container for operator functions in the share_a_ride problem.
    Operators are stateless functions that transform routes.
    """
    def __init__(
            self,
            operator: Callable,
            args: Dict[str, Any] = {}
        ):
        """
        Initialize the operator with problem instance and parameters.
        
        Params:
            - problem: ShareARideProblem instance
            - operator: Callable implementing the operator (e.g., destroy, build)
            - args: Arguments for the operator execution
                (e.g., max_remove, num_actions, T, seed, verbose)
        """
        self.operator = operator
        self.args = args or {}


    def apply(self, problem: ShareARideProblem, route: List[int]) -> List[int]:
        """
        Apply the operator to a given route.
        Params:
            - route: Input route to transform
        Returns:
            - res_route: The route after applying the operator
        """

        # Apply operator with problem instance and route
        res_route = self.operator(problem, route, **self.args)

        return res_route


    def __call__(self, **kwargs) -> List[int]:
        """
        Convenience method to apply operator directly.
        """
        return self.apply(**kwargs)
