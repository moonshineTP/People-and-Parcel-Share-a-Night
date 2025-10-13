# Initial VRPLIB and Solomon dataset curation to SARP format
**Author:** *moonshineTP*


This is the dataset for the **VRPLIB** and Solomon instances curated to fit the 
**Share-a-Ride Problem (SARP)** format. Refer from https://github.com/PyVRP/VRPLIB.


## Curation workflow summary
- Parse **VRPLIB** format (see: https://github.com/PyVRP/VRPLIB/tree/main/vrplib/parse)
- Parse **Solomon** format (see: https://github.com/PyVRP/VRPLIB/tree/main/vrplib/parse)
- For **VRPLIB** instances:
    - Retain only **CVRP** instances with **EDGE_WEIGHT_TYPE = EUC_2D** or **EXPLICIT**.
    - Remove instances with non-unique depot, or multiple depots.
    - Remove instances with even number of nodes (to ensure at least one passenger pickup-delivery pair).
- For **Solomon** instances:

- Compute pairwise Euclidean distances (if **EDGE_WEIGHT_TYPE = EUC_2D**), else use the explicit matrix.
- Sample passenger/parcel pairs from customers (excluding depot) with a controllable ratio.
- Sample parcel quantities, and vehicle capacities via **Poisson** distributions with described parameters.
- Write **SARP**, including newly defined sections: **PAIR_SECTION**, **VEHICLE_CAPACITY_SECTION**, **PARCEL_QUANTITY_SECTION**.
- Save curated instances in **.sarp** text format.

*Refer to ./data_description.txt for details on the VRPLIB format.*

*Refer to tests/data/data_description.txt for SARP format details.*


## Dataset folders usage summary
- **dataset/Vrp-Set-X**: VRPLIB instances for evaluation/comparison; input to curation.
- **dataset/cvrplib**: benchmark VRPLIB set.
- **dataset/lkh-3**: final evaluation test set.
- **tests/data**: small illustrative example and data description.