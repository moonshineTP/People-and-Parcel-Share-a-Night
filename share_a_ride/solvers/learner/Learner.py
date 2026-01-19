from typing import Dict, Any, Optional, Tuple, List

from share_a_ride.core.problem import ShareARideProblem
from share_a_ride.core.solution import Solution


class LearnerSolver:
    """
    Container for learning-based solvers in the share_a_ride problem.
    """

    def __init__(
        self,
        model: Any = None,
        args: Dict[str, Any] = {},
        hyperparams: Dict[str, Any] = {},
    ):
        """
        Initialize the learner with problem instance and model parameters.
        Params:
            - problem: ShareARideProblem instance to solve
            - model: Learning model (e.g., neural network, RL agent)
            - args: Additional arguments for training/inference
                (e.g. time_limit, verbose, seed)
            - hyperparams: Dictionary of model-specific hyperparameters
                (e.g. learning_rate, epochs, hidden_units, batch_size, etc.)
        """
        # Learner state
        self.model = model
        self.args = args or {}
        self.hyperparams = hyperparams or {}

        # Metadata
        self.name = "Learner (Not implemented)"
        self.is_trained = False
        self.training_history = {}

    def fit(
        self, training_data: Any = None, validation_split: float = 0.2, **kwargs
    ) -> Dict[str, Any]:
        """
        Train the learning model on provided data or through self-play.

        Params:
            - training_data: List of (problem, solution) pairs for supervised learning
                If None, should use reinforcement learning or self-play
            - validation_split: Fraction of data to use for validation
            - kwargs: Additional training arguments

        Returns:
            training_info: Dictionary with training statistics including:
                + loss: Training loss values
                + val_loss: Validation loss values (if applicable)
                + metrics: Other training metrics
                + time: Training time
        """
        return {}

    def predict(self, problem: ShareARideProblem, sol_state: Any = None) -> Any:
        """
        Use the trained model to generate a prediction for the given problem.
        (e.g. action, route, assignment, etc.)
        Params:
            - problem: ShareARideProblem instance to solve
            - sol_state: current solution state (if applicable)
        Returns:
            - prediction: Model's prediction output
        """
        return None

    def solve(
        self, problem: ShareARideProblem
    ) -> Tuple[Optional[Solution], Dict[str, Any]]:
        """
        Main solving method using the trained model and its functionalities.

        Returns:
            (solution, info): tuple where
                - solution: Best Solution object found (or None if none found)
                - info: Dictionary with statistics including:
                    + time: Total time taken
                    + status: "done" or "timeout" or "not_trained"
                    + model-specific metrics
        """
        return (None, {})

    def evaluate(
        self, test_data: List[Tuple[ShareARideProblem, Solution]]
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.

        Params:
            - test_data: List of (problem, solution) pairs for evaluation

        Returns:
            eval_metrics: Dictionary with evaluation metrics including:
                + avg_cost: Average solution cost
                + accuracy: Solution quality metrics
                + inference_time: Average time per prediction
        """
        return {}

    def tune(
        self,
        lb_hyperparams: Dict[str, Any],
        ub_hyperparams: Dict[str, Any],
        n_trials: int = 10,
        training_data: Optional[List[Tuple[ShareARideProblem, Solution]]] = None,
    ) -> Dict[str, Any]:
        """
        Hyperparameter tuning method for the learning model.

        Params:
            - lb_hyperparams: Lower bounds for hyperparameters
            - ub_hyperparams: Upper bounds for hyperparameters
            - n_trials: Number of tuning trials to perform
            - training_data: Training data for model fitting during tuning

        Returns:
            best_hyperparams: Dictionary of best found hyperparameters
        """
        try:
            pass
        except ImportError as exc:
            raise ImportError(
                "Optuna is required for hyperparameter tuning."
                " Please install it via 'pip install optuna'."
            ) from exc

        return {}

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Params:
            - filepath: Path where model should be saved
        """
        pass

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Params:
            - filepath: Path to saved model
        """
        pass
