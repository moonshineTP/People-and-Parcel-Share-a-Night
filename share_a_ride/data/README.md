**Title**: **SARP data**  
**Version**: **1.0**  
**Maintainer**: **moonshineTP**  
**Format type**: **Extended VRPLIB**

---

**1. OVERVIEW**  

This dataset represents large-scale instances of the *Share-a-Ride Problem (SARP)*
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
These instances are curated from existing datasets:


---

**6. OUTPUT FORMAT (REFERENCE)**  
Like `.sol` in **VRPLIB/TSPLIB**:  
- **Route #i:** `v1 v2 … vj` (NO depot node appeared)  
- **Cost:** `<int/float>` (max route cost)

---
**7. USAGE**  
Benchmark for:  
- *Share-a-Ride Problem (SARP)*  
- *Matheuristics*, *ALNS*, *DRL-based*, *MILP solvers*

Use TSPLIB/VRPLIB readers and extend for custom sections.

---
END OF FILE
