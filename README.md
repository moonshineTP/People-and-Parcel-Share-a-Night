# People-and-Parcel-Share-a-Night
A variety of methods implemented to tackle the -- People and Parcel Share-a-Ride -- problem in Operations Research

---

## Overview
This repository contains data, benchmarks, models, and algorithms for solving the People and Parcel Share-a-Ride problem. This problem involves optimizing routes for vehicles that can carry both passengers and parcels, aiming to minimize operation cost differences between each vehicle.

---

## Main components
The main top directory is share_a_ride, which contains the following submodules:
- **Core**: Definitions of the problem and solution objects, including methods for evaluating and manipulating them.
- **Data**: A collection of benchmark datasets in Extended VRPLIB format, adapted for the Share-a-Ride problem with helper scripts for loading, parsing, attempting solutions, and summarizing results.
- **Precurated**: A collection of Vehicle Routing Problem datasets, used to create standardized new SARP instances through specific curating scripts.
- **Solvers**: A variety of algorithms and heuristics for solving the Share-a-Ride problem, including exact methods, metaheuristics, and learning-based approaches.
- **App**: A tkinter application for visualizing solutions and analyzing performance metrics.

Other scripts include main.py for running experiments, oj.py for online judging, inliner.py for inlining imports, submission.py for inlined submissions, and tests/ for unit tests.

---

## Problem Definition
There are several variants and definitions of the People and Parcel Share-a-Ride problem throughout the literature. In this repository, we focus on the variant close to the original problem proposed by Li et al. (2014), which involves a fleet of vehicles starting from a single depot, serving a set of passenger and parcel requests, and returning to that same depot. Each vehicle has a limited capacity of load and can serve only one passenger at a time. Formally:

**Given:**  
- **K** taxis (vehicles) located at a depot (node `0`).  
- **N** passenger requests and **M** parcel requests.

**Each request consists of two nodes:**  
- *Passenger* `i` (1 ≤ i ≤ N):  
    - **pickup node** = `i`  
    - **dropoff node** = `i + N + M`  
    - *constraint:* served *directly* (pickup → dropoff with no intermediate stops).  
- *Parcel* `j` (1 ≤ j ≤ M):  
    - **pickup node** = `j + N`  
    - **dropoff node** = `j + 2N + M`  
    - each parcel `j` has quantity *q[j]* (load/weight).

Each vehicle **k** has maximum capacity *Q[k]*.  
Distance between nodes *i* and *j* is *D(i,j)*. All routes start and end at the depot (`0`).

**Objective (benchmark):**  
Minimize the **maximum total route cost** among all taxis, subject to direct-ride and capacity constraints.

This objective aims to balance the workload among vehicles, ensuring that no single vehicle is disproportionately burdened compared to others. Most existing works on SARP focus on minimizing the total operation cost across all vehicles, which can lead to unbalanced solutions where some vehicles are heavily loaded while others are underutilized. By minimizing the maximum route cost, we promote fairness and efficiency in the allocation of resources.

(Note that N, M, K, q, Q, D is hard-coded as its accorded variable names in the source code of the problem and solution classes. Refer to the code documentation for more details. Hope that it does not confuse you too much.)

---

## Prerequisites
This codebase doesnt need all the packages at once. Depending on the module you want to use, you may need to install its additional packages.
The main dependencies are:
- **Python 3.8+** (prefer `Cpython` for better compatibility with all the libraries, `Pypy3` if you only need high performance)
- **Matplotlib** and its tkinter GUI backend for dashboard and visualization.
- **Pandas** and its dependencies for data loading and processing.
- **NetworkX** for some algorithms that aims to use a stricter graph representations.
- **Gurobipy** for Mixed Integer Linear Programming methods (requires a Gurobi license).
- **OR-Tools** for Constraint Programming methods.
- **Optuna** for hyperparameter optimization (optional).
- **DL frameworks** coming soon.

---

## Repository Structure

```
People-and-Parcel-Share-a-Night/
├── __init__.py
├── description.txt
├── inliner.py
├── LICENSE
├── main.py
├── oj.py
├── README.md
├── requirements.txt
├── submission.py
└── share_a_ride/
    ├── __init__.py
    ├── app/
    │   ├── app.py
    │   ├── dashboard.py
    │   ├── visualizer.py
    ├── core/
    │   ├── problem.py
    │   ├── solution.py
    │   └── utils/
    │       ├── generator.py
    │       ├── helper.py
    ├── data/
    │   ├── example.sarp
    │   ├── executor.py
    │   ├── loader.py
    │   ├── parser.py
    │   ├── README.md
    │   ├── router.py
    │   ├── summarizer.py
    │   ├── benchmark/
    │   │   ├── Golden/
    │   │   ├── Li/
    │   │   ├── Pyvrp/
    │   │   └── Solomon/
    │   ├── sanity/
    │   │   ├── Exact/
    │   │   └── H/
    │   ├── test/
    │   │   └── Tai/
    │   ├── train/
    │   │   └── Cvrplib/
    │   └── val/
    │       └── CMT/
    ├── precurated/
    │   ├── curater_solomon.py
    │   ├── curater_vrplib.py
    │   ├── description.txt
    │   ├── README.md
    │   ├── utils.py
    │   ├── CMT/
    │   │   └── README.txt
    │   ├── Cvrplib/
    │   │   └── README.txt
    │   ├── Golden/
    │   │   └── README.txt
    │   ├── Li/
    │   │   └── README.txt
    │   ├── Pyvrp/
    │   │   └── README.txt
    │   ├── Solomon/
    │   │   └── README.txt
    │   └── Tai/
    │       └── README.txt
    ├── solvers/
    │   ├── algo/
    │   ├── learner/
    │   ├── operator/
    │   ├── policy/
    │   └── utils/
    └── tests/
```

---

## Dataset
There seems to be no standardized datasets specifically designed for the People and Parcel Share-a-Ride problem. To address this, we have curated a collection of benchmark instances adapted from existing Vehicle Routing Problem datasets. These instances are curated from existing datasets, some of them include Li et al., Solomon et al., Talliard et al.,... The final curated datasets are stored in the `share_a_ride/data/benchmark/` directory with the data format extended from VRPLIB/TSPLIB format famous from the literature to accommodate the SARP requirements.

All the original data sources are properly cited in the folder of each dataset.

The modifications made to adapt them to the SARP context, including adding parcel requests and adjusting vehicle capacities, are detailed in the share_a_ride/precurated directory.

---

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/moonshineTP/People-and-Parcel-Share-a-Night.git
   cd People-and-Parcel-Share-a-Night
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. For optional solvers / API (Gurobi, OR-Tools), install them separately as needed.

4. Feel free to modify and extend the codebase as per your requirements. There should be not much package conflict as this scale.

---

## Usage

- **Dashboard**: Run `python share_a_ride/app/app.py` to launch the Tkinter dashboard for visualization and analysis.
- **Experiments**: Use `main.py` to run solver experiments on datasets.
- **Online Judging Writing**: Use `oj.py` for online judge submission overwrite. You still need to resolve import from `share_a_ride/` right? That is what inliner.py is for. Run `python inliner.py` to create an inlined version for submission (remember to check the path of the output file in inliner.py).
- **Online Judging Submission**: Use `submission.py` to run the inlined submission file. It is not perfect yet, but it should work in most cases with some minor adjustments and deduplication.
- **Tests**: Run `python -m pytest tests/` to execute unit tests.

Refer to individual module READMEs for detailed usage.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes and add tests.
4. Submit a pull request.

You should also make sure your code follows best practices, including adhering to PEP 8 style guidelines, passing all existing tests, and adding new tests for any new functionality. For that, you need to import some linters and type checkers. I recommend `pylint` for style checking and `basedpyright` for type checking. Also remember to include docstrings.

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## References

- Li, B., et al. (2014). The Share-a-Ride Problem: People and Parcels Sharing Taxis. (Main paper)
- Other datasets: See citations in respective folders under `share_a_ride/precurated/`.

For more details, refer to the code documentation and papers cited.

---

## Acknowledgements
This work is just a class project. In details, it's for the course IT4663 (Planning and Optimization) of Hanoi University of Science and Technology (HUST). As such, the problem definition and aim may not be general and helpful enough for some real usage/ literature. Special thanks to the course instructors and peers for their support and feedback.

---

## Citings
Currently we dont have a proper paper to cite for this repository. If you use this codebase in your research, please cite the github repository:
```@
misc{moonshineTP_share_a_ride,
  author = {moonshineTP},
  title = {People-and-Parcel-Share-a-Night: A Repository for the People and Parcel Share-a-Ride Problem},
  year = {2025},
  url = {https://github.com/moonshineTP/People-and-Parcel-Share-a-Night},
}
```

---