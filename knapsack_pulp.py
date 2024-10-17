import pulp

def KS_PULP(values, weights, capacity):
    num_items = len(values)
    # Define the LP problem
    prob = pulp.LpProblem("Knapsack Problem", pulp.LpMaximize)

    # Decision variables: x_i in {0,1}
    x = [pulp.LpVariable(f'x_{i}', cat='Binary') for i in range(num_items)]

    # Objective function: Maximize total value
    total_value = pulp.lpSum(values[i] * x[i] for i in range(num_items))
    prob += total_value

    # Capacity constraint
    total_weight = pulp.lpSum(weights[i] * x[i] for i in range(num_items))
    prob += (total_weight <= capacity)

    # Solve the problem
    prob.solve()

    # Extract the results
    selected_items = [i + 1 for i in range(num_items) if pulp.value(x[i]) == 1]
    total_value = sum(values[i] for i in range(num_items) if pulp.value(x[i]) == 1)
    total_weight = sum(weights[i] for i in range(num_items) if pulp.value(x[i]) == 1)

    return selected_items, total_value, total_weight

if __name__ == "__main__":
    v = [10, 12, 8, 5, 8, 5, 6, 7, 6, 12, 8, 8, 10, 9, 8, 3, 7, 8, 5, 6]
    w = [6, 7, 7, 3, 5, 2, 4, 5, 3, 9, 8, 7, 8, 6, 5, 2, 3, 5, 4, 6]
    K = 50

    selected_items, total_value, total_weight = KS_PULP(
        v, w, K
    )

    print("Linear Programming Solution:")
    print("Selected item indices:", selected_items)
    print("Total value:", total_value)
    print("Total weight:", total_weight)
