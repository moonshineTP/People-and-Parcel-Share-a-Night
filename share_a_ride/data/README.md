**Title**: **SARP data**  
**Version**: **1.0**  
**Maintainer**: **moonshineTP**  
**Format type**: **Extended VRPLIB**

---

**1. OVERVIEW**  

This dataset represents small-scale to large-scale instances of the *Share-a-Ride Problem (SARP)*

The format extends the well-known **TSPLIB/VRPLIB** structure to accommodate the unique requirements of SARP, which involves coordinating multiple taxis to serve both passenger and parcel requests with specific constraints.

---

**2. PROBLEM DESCRIPTION**  
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
Distance between nodes *i* and *j* is *d(i,j)*. All routes start and end at the depot (`0`).

**Objective (benchmark):**  
Minimize the **maximum total route cost** among all taxis, subject to direct-ride and capacity constraints.

---

**3. FILE STRUCTURE**  
Each `<name>.sarp` file follows **TSPLIB** header syntax.

**Required sections:**  
- `NAME : <string>`                     # instance name  
- `COMMENT: <string>`                   # comment  
- `TYPE : SARP`                         # problem type  
- `DIMENSION : <integer>`               # nodes = 2N + 2M + 1  
- `EDGE_WEIGHT_TYPE : EXPLICIT`         # {EXPLICIT, EUC_2D}  
- `EDGE_WEIGHT_FORMAT : FULL_MATRIX`    # optional, only support FULL_MATRIX
- `CAPACITY : <integer>`                # vehicle capacity (if uniform)
- `EDGE_WEIGHT_SECTION`  
        ```  
        D lines of D integers each  
        ```  
- `EOF_EDGE_WEIGHT_SECTION`  

- `NODE_COORD_SECTION`  
        ```  
        D lines of "<id> <x> <y>"  
        ```  
- `EOF_NODE_COORD_SECTION`  

- `NODE_TYPE_SECTION`  
        ```
        id  node_id  type  
        ```  
    type ∈ {0=DEPOT,1=PASS_PICKUP,2=PARC_PICKUP,3=PASS_DROPOFF,4=PARC_DROPOFF}  
- `EOF_NODE_TYPE_SECTION`

- `PAIR_SECTION`  
        ```
        id  pickup_node  category  drop_node  
        ```  
    category ∈ {P=passenger, L=parcel}  
- `EOF_PAIR_SECTION`

- `VEHICLE_CAPACITY_SECTION`  
        ```
        id  vehicle_id  capacity  
        ```  
- `EOF_VEHICLE_CAPACITY_SECTION`

- `PARCEL_QUANTITY_SECTION`  
        ```
        id  parcel_node_id  quantity  
        ```  
- `EOF_PARCEL_QUANTITY_SECTION`

- `DEPOT_SECTION`  
        ```
        depot_node_id  
        ```  
- `EOF_DEPOT_SECTION`

All sections end with `EOF_<SECTION>_SECTION`. 
File ends with **EOF**.

---

**4. DATA UNITS AND CONVENTIONS**  
**Indexing:**  
- File IDs are 1-based; internal processing can be 0-based.
- Depot node ID should be 1 across all instances.

**Units:**  
- Distances: integer/float; unit = meters (if unspecified).  
- Parcel quantities *q_j* and capacities *Q_k*: integer load units (e.g., kg).

**Distance matrix:**  
- FULL_MATRIX over 0..D-1; d(i,i)=0; d(i,j)≥0; should expect it is asymmetric.

**Service:**  
- Each node visited exactly once
- Passengers: only one onboard, not necessarily direct.
- Parcels: multiple onboard, as long as capacity constraints are met; intermediate stops allowed.

**Parsing:**  
- Numeric fields: base-10 integers, whitespace-separated.  
- Section names/EOF markers: uppercase.  
- UTF-8 text.  
- 32-bit ints suffice.

**Scope:**  
- No travel/service times or time windows unless extended.

---

**5. RELATION TO SOURCE DATA**  
These instances are curated from existing datasets, some of them include Li et al., Solomon et al., Talliard et al.,...
Modifications is made to adapt them to SARP context, including adding parcel requests and adjusting vehicle capacities.
The detail of modifications is mentioned in share_a_ride/precurated
All original data sources are properly cited in the folder of each dataset.

---

**6. OUTPUT FORMAT (REFERENCE)**  
Like `.sol` in **VRPLIB/TSPLIB**:  
- **Route #i:** `v1 v2 … vj` (NO depot node appeared)  
- **Cost:** `<int/float>` (max route cost)

---
**7. USAGE**  
Benchmark for:  
- *Exact methods*: Exhaustive search, Branch-and-Bound, MILP solvers, Branch-and-Cut.
- *Tree search methods*: Beam Search, Monte Carlo Tree Search.
- *Local search methods*: Tabu Search, Variable Neighborhood Search.
- *Population-based methods*: Simulated Annealing, Adaptive Large Neighborhood Search, Genetic Algorithms.

8. Benchmarking Procedure
The benchmarking procedure involves the following steps:
- Configure a solver. File paths are resolved via path_router.
- Run attempts:
        - attempt_instance(dataset, instance_name): parses the <name>.sarp, calls solver.solve, and appends one row to the dataset attempts CSV.
                - attempt_id: auto-increment (1-based; header written if the file is empty).
                - timestamp: UTC ISO-8601 without microseconds.
                - solver fields: solver name, seed and time_limit from solver.args; hyperparams as JSON.
                - results: status from solver info (done/timeout/error), time from info, cost = sol.max_cost or None.
                - info JSON: a pruned copy of solver info (status and time removed).
                - gap% (in-memory): computed only if both current cost and prior best_cost exist.
        - attempt_dataset(dataset): iterates all .sarp instances in the dataset and calls attempt_instance for each; returns (solutions, gaps, textual summary).
- Summarize results:
        - summarize_dataset(dataset): enumerates all instances in the dataset folder and calls summarize_instance per instance.
        - summarize_instance(dataset, inst_name): reads all rows for the instance from the attempts CSV, then writes/updates a single row in the dataset scoreboard CSV:
                - Counts: num_attempts, successful_attempts (status == done).
                - Best: min cost among done attempts; captures attempt_id, timestamp, solver, args (seed/time_limit only), hyperparams, elapsed_time.
                - Improvement vs previous best in the scoreboard: cost_improvement and percentage_improvement (rounded to 2 decimals); sets note = "improved" if improved.
                - If no successful attempt exists: best_* and improvement fields are None.
        - The scoreboard file must exist (can be empty); header is written if empty.

- Loading data for Dashboard and Visualization:
        - Use dataloader.py to load attempts and scoreboard CSVs into DataFrames.
        - The GUI in gui.py then visualizes:
                - Instance-wise performance over attempts (costs, gaps, times).
                - Solver-wise comparisons (best costs, success rates).
        - Allows filtering by dataset, solver, hyperparameters.

---


**9. DATASET STRUCTURE**
This folder classifies datasets into subfolders based on their usage in the research:
- `sanity`: small instances for initial testing and debugging.
- `train`: medium-sized instances for training.
- `val`: validation instances for hyperparameter tuning.
- `test`: large-scale instances for evaluation, specifically for learning-augmented methods.
- `benchmark`: comprehensive set of instances for benchmarking methods of all types across scales.


**10. ADDITIONAL CONSIDERATIONS**

**Notes:**
- attempt_instance creates the attempts CSV header if the file is empty; attempt_id = number of existing lines (header excluded).
- summarize_instance requires the scoreboard file to exist; create an empty file first if needed.
- Gaps are reported during execution only when a prior best_cost exists for the instance.

**Citations:**
If you use this dataset in your research, please cite the github repository:
```@misc{moonshineTP_share_a_ride,
  author = {moonshineTP},
  title = {Share-a-Ride Problem (SARP) Dataset and Benchmark},
  year = {2025},
  url = {https://github.com/moonshineTP/People-and-Parcel-Share-a-Night/tree/main/share_a_ride/data},
}
```

**Contributions:**
Contributions and improvements to this dataset and benchmark are welcome. Please submit issues or pull requests on
the GitHub repository.

---

END OF FILE