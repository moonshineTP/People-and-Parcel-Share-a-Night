from typing import Any, Callable, Dict, Optional

from share_a_ride.core.problem import ShareARideProblem


class Policy():
    """
    Container for policy-based solvers/learners in the share_a_ride problem.
    Support rule-based, probabilistic, and learning-based policies.
    """

    def __init__(
            self,
            policy: Callable[..., float],
            args: Optional[Dict[str, Any]] = None,
            hyperparams: Optional[Dict[str, Any]] = None
        ):
        """
        Initialize the policy solver with policy function.
        
        Params:
        - policy: Callable implementing the policy function
        - args: Additional arguments for the policy
            (e.g. time_limit, verbose, seed)
        - hyperparams: Dictionary of policy-specific hyperparameters
            (e.g. temperature, exploration_rate, model parameters, etc.)
        """
        self.policy = policy
        self.args = args or {}
        self.hyperparams = hyperparams or {}

        self.is_trained = False # For learning-based policies


    def execute(
            self,
            problem: ShareARideProblem,
            state: Any
        ) -> Any:
        """
        Execute the policy function on the given state.
        
        Params:
        - problem: ShareARideProblem instance
        - state: Current state representation for the policy
            (e.g. current routes, unassigned requests, etc.)
            This will be implemented later.
        
        Returns:
        (action, info): tuple where
            - action: Selected action from the policy
            - info: Dictionary with execution statistics
        """
        action = self.policy(problem, state, **self.args, **self.hyperparams)

        return action


    def tune(
            self,
            lb_hyperparams: Dict[str, Any],
            ub_hyperparams: Dict[str, Any],
            n_trials: int = 10,
        ) -> Dict[str, Any]:
        """
        Hyperparameter tuning method for the policy.
        
        Params:
        - lb_hyperparams: Lower bounds for hyperparameters.
            If the hyperparameter is non-numeric, this should be a list of possible values.
        - ub_hyperparams: Upper bounds for hyperparameters.
            If the hyperparameter is non-numeric, this should be a list of possible values.
        - n_trials: Number of tuning trials to perform
        - kwargs: Additional tuning arguments
        
        Returns:
        best_hyperparams: Dictionary of best found hyperparameters
        """
        pass


    def save(self, filepath: str) -> None:
        """
        Save the policy configuration to a file.
        Params:
        - filepath: Path to save the policy configuration
        """
        pass


    def load(self, filepath: str) -> None:
        """
        Load the policy configuration from a file.
        Params:
        - filepath: Path to load the policy configuration from
        """
        pass