'''
* Genetic Algorithm for the Knapsack Problem (1)
* CC2017 - Modelación y Simulación, 2024
* Samuel Chamalé, Adrian Rodriguez y Daniel Gómez
'''

import random

def KS_GA(values, weights, capacity, params):
    """
    Solves the Knapsack problem using a genetic algorithm.

    Parameters:
    - values: List of item values.
    - weights: List of item weights.
    - capacity: Maximum weight capacity of the knapsack.
    - params: Dictionary of GA parameters.

    Returns:
    - best_solution: Binary list indicating selected items.
    - best_value: Total value of the best solution.
    - best_weight: Total weight of the best solution.
    """
    # GA Parameters
    N = params.get('population_size', 100)
    s = params.get('selection_rate', 0.5)
    c = params.get('crossover_rate', 0.7)
    m = params.get('mutation_rate', 0.01)
    max_iterations = params.get('max_iterations', 1000)
    t_size = params.get('tournament_size', 3)
    elitism = params.get('elitism', True)
    penalty_coefficient = params.get('penalty_coefficient', 1000)

    num_items = len(values)

    # Helper functions
    def initialize_population():
        population = []
        for _ in range(N):
            individual = [random.randint(0, 1) for _ in range(num_items)]
            population.append(individual)
        return population

    def evaluate(individual):
        total_value = sum(v * x for v, x in zip(values, individual))
        total_weight = sum(w * x for w, x in zip(weights, individual))
        if total_weight <= capacity:
            fitness = total_value
        else:
            fitness = total_value - penalty_coefficient * (total_weight - capacity)
        return fitness, total_value, total_weight

    def tournament_selection(population, fitnesses):
        selected = []
        current_population_size = len(population)
        num_selected = int(s * current_population_size)
        for _ in range(num_selected):
            competitors = random.sample(range(current_population_size), t_size)
            best = max(competitors, key=lambda idx: fitnesses[idx][0])
            selected.append(population[best])
        return selected

    def crossover(parent1, parent2):
        if random.random() < c:
            child1, child2 = [], []
            for gene1, gene2 in zip(parent1, parent2):
                if random.random() < 0.5:
                    child1.append(gene1)
                    child2.append(gene2)
                else:
                    child1.append(gene2)
                    child2.append(gene1)
            return child1, child2
        else:
            return parent1[:], parent2[:]

    def mutate(individual):
        for i in range(num_items):
            if random.random() < m:
                individual[i] = 1 - individual[i]
        return individual

    def repair(individual):
        while True:
            total_weight = sum(w * x for w, x in zip(weights, individual))
            if total_weight <= capacity:
                break
            ones_indices = [i for i, gene in enumerate(individual) if gene == 1]
            if not ones_indices:
                break
            idx = random.choice(ones_indices)
            individual[idx] = 0
        return individual

    # Initialization
    population = initialize_population()
    fitnesses = [evaluate(ind) for ind in population]
    best_solution = max(zip(population, fitnesses), key=lambda x: x[1][0])
    best_individual = best_solution[0][:]
    best_fitness, best_value, best_weight = best_solution[1]

    # Main GA loop
    for iteration in range(max_iterations):
        # Selection
        selected_parents = tournament_selection(population, fitnesses)

        # Crossover and Mutation
        offspring = []
        while len(offspring) < N:
            parent1, parent2 = random.sample(selected_parents, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            child1 = repair(child1)
            child2 = repair(child2)
            offspring.extend([child1, child2])

        # If offspring exceeds N, truncate the list
        offspring = offspring[:N]

        # Evaluate Offspring
        offspring_fitnesses = [evaluate(ind) for ind in offspring]

        # Replacement with Elitism
        if elitism:
            # Find the worst individual in offspring
            worst_idx = min(range(len(offspring_fitnesses)), key=lambda idx: offspring_fitnesses[idx][0])
            # Replace it with the best individual from the current population
            offspring[worst_idx] = best_individual
            offspring_fitnesses[worst_idx] = (best_fitness, best_value, best_weight)

        population = offspring
        fitnesses = offspring_fitnesses

        # Update Best Solution
        current_best = max(zip(population, fitnesses), key=lambda x: x[1][0])
        if current_best[1][0] > best_fitness:
            best_individual = current_best[0][:]
            best_fitness, best_value, best_weight = current_best[1]

    # Final Best Solution
    selected_items = [i for i, gene in enumerate(best_individual) if gene == 1]
    return selected_items, best_value, best_weight


if __name__ == "__main__":
    v = [10, 12, 8, 5, 8, 5, 6, 7, 6, 12, 8, 8, 10, 9, 8, 3, 7, 8, 5, 6]
    w = [6, 7, 7, 3, 5, 2, 4, 5, 3, 9, 8, 7, 8, 6, 5, 2, 3, 5, 4, 6]
    K = 50

    # GA parameters
    params = {
        'population_size': 200,
        'selection_rate': 0.5,
        'crossover_rate': 0.8,
        'mutation_rate': 0.02,
        'max_iterations': 1000,
        'tournament_size': 5,
        'elitism': True,
        'penalty_coefficient': 1000
    }

    # Run GA
    selected_items, total_value, total_weight = KS_GA(v, w, K, params)

    # Output the results
    print("Genetic Algorithm Solution:")
    print("Selected item indices:", [i + 1 for i in selected_items])  # Adjusting index to match item numbering
    print("Total value:", total_value)
    print("Total weight:", total_weight)
