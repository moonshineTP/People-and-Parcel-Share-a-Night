"""
Module for defining data classes used in data processing in share_a_ride/data.
These classes encapsulate data related to datasets, instances, attempts, and solutions.
"""
from enum import Enum
from dataclasses import dataclass




# =============== Enums and Data Classes ================
class Action(Enum):
    """
    Enum for different actions related to data and file management.
    """
    READALL = "readall"
    READFILE = "readfile"
    RECORD = "record"
    SOLVE = "solve"
    SUMMARIZE = "summarize"


    @property
    def extension(self) -> str:
        """
        Get the file extension associated with the action.
        """
        extensions = {
            Action.READFILE: ".sarp",
            Action.SOLVE: ".sol",
        }
        return extensions.get(self, "")




class Purpose(Enum):
    """
    Enum for dataset purposes.

    There are five purposes:
    - SANITY: small instances for sanity checks and debugging
    - TRAIN: training datasets for learning-based methods
    - VAL: validation datasets for hyperparameter tuning
    - TEST: test datasets for final evaluation
    - BENCHMARK: benchmark datasets for comparing against literature 
    """
    SANITY = "sanity"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    BENCHMARK = "benchmark"




@dataclass
class DatasetInfo:
    """
    Data class to hold dataset information.

    Attributes:
    - name: name of the dataset
    - purpose: purpose of the dataset
    """
    name: str
    purpose: Purpose




class Dataset(Enum):
    """
    Enum for different datasets used in share_a_ride.
    """
    HUST = DatasetInfo("Hust", Purpose.SANITY)
    EXACT = DatasetInfo("Exact", Purpose.SANITY)
    LI = DatasetInfo("Li", Purpose.BENCHMARK)
    SOLOMON = DatasetInfo("Solomon", Purpose.TEST)
    PYVRP = DatasetInfo("Pyvrp", Purpose.BENCHMARK)
    GOLDEN = DatasetInfo("Golden", Purpose.BENCHMARK)
    CVRPLIB = DatasetInfo("Cvrplib", Purpose.TRAIN)
    CMT = DatasetInfo("CMT", Purpose.VAL)
    TAI = DatasetInfo("tai", Purpose.TEST)


    @classmethod
    def from_str(cls, name: str) -> 'Dataset':
        """
        Get Dataset enum member from string name.
        """
        for item in cls:
            if item.value.name.lower() == name.lower():
                return item

        raise ValueError(f"Unknown dataset: {name}")




# ================ CSV Columns ================
ATTEMPT_COLUMNS = [
    'attempt_id',
    'timestamp',
    'dataset',
    'instance',
    'solver',
    'seed',
    'time_limit',
    'hyperparams',
    "status",
    'elapsed_time',
    'cost',
    'info',
    'note'
]


SCOREBOARD_COLUMNS = [
    'dataset',
    'instance',
    'num_attempts',
    'successful_attempts',
    'best_cost',
    'best_attempt_id',
    'best_timestamp',
    'best_solver',
    'best_solver_args',
    'best_solver_hyperparams',
    'best_time_taken',
    'cost_gap',
    'pct_gap',
    'note'
]
