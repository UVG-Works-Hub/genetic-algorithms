'''
* Genetic Algorithm for the Knapsack Problem (1)
* CC2017 - Modelación y Simulación, 2024
* Samuel Chamalé, Adrian Rodriguez y Daniel Gómez
'''

import random

def KP_GA(v, w, K, N=50, s=0.2, c=0.8, m=0.01, maxI=1000, penalty=1000, t_size=3, elitism=True):
    '''
    This function solves the Knapsack Problem using a Genetic Algorithm.

    Parameters:
    - v: list of values of the items
    - w: list of weights of the items
    - K: maximum weight allowed
    - N: number of individuals in the population
    - s: selection rate
    - c: crossover rate
    - m: mutation rate
    - maxI: maximum number of iterations
    - penalty: penalty coefficient
    - t_size: tournament size
    - elitism: whether to use elitism or not

    Returns:
    - selected: list of selected items
    - best_value: value of the best solution
    - best_weight: weight of the best solution
    '''
    def populate():
        '''
        Initialize the population with random individuals.

        Returns:
        - population: list of individuals
        '''
        population = []
        for _ in range(N):
            individual = [random.randint(0,1) for _ in range(len(v))]
            population.append(individual)
        return population

    def evaluate(individual):
        '''
        For a given individual, evaluate its fitness.
        This means calculating the total value and total weight of the items selected.

        Parameters:
        - individual: list of 0s and 1s representing the items selected

        Returns:
        - fitness: value of the fitness function
        - total_value: total value of the items selected
        - total_weight: total weight of the items selected
        '''
        total_value = sum(v * x for v, x in zip(v, individual)) # A binary product
        total_weight = sum(w * x for w, x in zip(w, individual)) # Again, a binary product

        if total_weight <= K: # If the weight is less than the maximum allowed
            fitness = total_value
        else:
            fitness = total_value - penalty * (total_weight - K)
        return fitness, total_value, total_weight

    def tournament_selection(population, fitnesses):
        '''
        Selects the parents for the next generation using tournament selection.

        Parameters:
        - population: list of individuals
        - fitnesses: list of fitness values for each individual

        Returns:
        - selected_parents: list of selected parents
        '''
        selected = []
        for _ in range(int(s * N)): # Select s * N parents
            fighters = random.sample(range(N), t_size) # Randomly select t_size individuals
            best = max(fighters, key=lambda idx: fitnesses[idx][0]) # Select the best one
            selected.append(population[best])
        return selected

    def crossover(p1, p2):
        '''
        Based on the crossover rate, create two children from two parents.

        Parameters:
        - p1: parent 1
        - p2: parent 2

        Returns:
        - c1: child 1
        - c2: child 2
        '''
        if random.random() < c: # If the crossover rate is met XD
            c1, c2 = [], [] # Children
            for g1, g2 in zip(p1, p2): # For each gene in the parents
                if random.random() < 0.5: # Randomly select one of the genes
                    c1.append(g1)
                    c2.append(g2)
                else:
                    c1.append(g2)
                    c2.append(g1)
            return c1, c2
        else:
            return p1[:], p2[:] # If the crossover rate is not met, return the parents :v

    def mutate(individual):
        '''
        Based on the mutation rate, flip the genes of the individual.

        Parameters:
        - individual: list of 0s and 1s

        Returns:
        - individual: mutated individual
        '''
        for i in range(len(v)): # For each gene in the individual
            if random.random() < m: # If the mutation rate is met
                individual[i] = 1 - individual[i] # Flip the gene
        return individual

    def repair(individual):
        '''
        Here we repair the individual if the total weight exceeds the maximum allowed.

        Parameters:
        - individual: list of 0s and 1s

        Returns:
        - individual: repaired individual
        '''
        while True:
            total_weight = sum(w * x for w, x in zip(w, individual))
            if total_weight <= K:
                break
            ones_indices = [i for i, gene in enumerate(individual) if gene == 1]
            if not ones_indices:
                break
            idx = random.choice(ones_indices)
            individual[idx] = 0
        return individual

    # Initialization
    population = populate() # Initial population
    fitnesses = [evaluate(ind) for ind in population] # Fitness of each individual
    best_solution = max(zip(population, fitnesses), key=lambda x: x[1][0]) # Based on fitness
    best_individual = best_solution[0][:] # Best individual, using : to copy the list
    best_fitness, best_value, best_weight = best_solution[1] # Best fitness, value and weight

    # Main GA loop
    for iteration in range(maxI):
        # Selection
        selected_parents = tournament_selection(population, fitnesses)

        # Crossover and Mutation
        offspring = [] # New generation
        for i in range(0, len(selected_parents), 2):
            p1, p2 = selected_parents[i], selected_parents[(i+1) % len(selected_parents)] # Parents
            c1, c2 = crossover(p1, p2) # Children
            c1 = mutate(c1)
            c2 = mutate(c2)
            c1 = repair(c1)
            c2 = repair(c2)
            offspring.extend([c1, c2])

        # Evaluate Offspring
        offspring_fitnesses = [evaluate(ind) for ind in offspring]

        # Replacement
        if elitism:
            # Keep the best individual from the current generation
            worst_idx = min(range(len(offspring_fitnesses)), key=lambda idx: offspring_fitnesses[idx][0])
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



