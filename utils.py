

def compute_tree_complexity(nested_dict: dict) -> dict:
    """
    Computes a few metrics for a nested dictionary structure,
    which simulates a directory tree with sub-folders as children.
    """
    total_nodes = 0
    max_depth = 0
    total_branches = 0
    leaf_count = 0

    def dfs(node, depth):
        nonlocal total_nodes, max_depth, total_branches, leaf_count
        total_nodes += 1
        max_depth = max(max_depth, depth)

        if isinstance(node, dict) and node:
            # number of sub-branches is the number of keys
            children_count = len(node.keys())
            total_branches += children_count
            for child in node.values():
                dfs(child, depth + 1)
        else:
            leaf_count += 1

    # Run DFS from the root
    dfs(nested_dict, 1)

    if total_nodes > 1:
        avg_branching_factor = total_branches / (total_nodes - 1) 
    else:
        avg_branching_factor = 0

    return {
        "Total Nodes": total_nodes,
        "Max Depth": max_depth,
        "Average Branching Factor": avg_branching_factor,
        "Leaf Count": leaf_count
    }


# Example usage:
if __name__ == "__main__":
    
    
    # example_structure = {
    #     "a": {
    #         "b": {}, 
    #         "c": {}
    #     },
    #     "d": {
    #         "e": {},
    #         "f": {}
    #     }
    # }

    import json
    example_structure = json.loads(open(r"C:\Users\Daniil Anisimov\git\bioptic-insights\dan\data\trials\obesity\all obesity trials\SHORT___(AREA[Condition] (obesity OR overweight NOT diabetes) AND (AREA[InterventionType] DRUG)).json").read())
    
    # compute the metrics for the example structure
    metrics = compute_tree_complexity(example_structure)
    print("Metrics for example structure:", metrics)