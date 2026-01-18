import os
import re

def list_pyvrp_instances():
    # Define the relative path to the Pyvrp directory
    # Based on workspace structure: share_a_ride/data/benchmark/Pyvrp
    base_dir = os.getcwd()
    pyvrp_dir = os.path.join(base_dir, "share_a_ride", "data", "benchmark", "Li")
    
    if not os.path.exists(pyvrp_dir):
        print(f"Directory not found at: {pyvrp_dir}")
        return

    # specific suffix for these instances appear to be .sarp based on the file attachment info
    # usually instance names are the filenames without extensions
    instance_names = []
    
    try:
        files = os.listdir(pyvrp_dir)
        for f in files:
            if f.endswith(".sarp"):
                # Remove the extension to get the instance name
                instance_name = os.path.splitext(f)[0]
                instance_names.append(instance_name)
        
        # Sort the list for better readability
        instance_names.sort(key=lambda f: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', f)])
        
        print(instance_names)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_pyvrp_instances()
