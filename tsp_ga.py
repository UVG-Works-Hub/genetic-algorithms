import math
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import multiprocessing
from copy import deepcopy
import os  # Added for directory operations

# Constants
INT_MAX = float('inf')  # Represents infinity for unreachable paths

# Configuration Parameters
POPULATION_SIZE = 250    
GENERATIONS = 300      
MUTATION_RATE = 0.04   
ELITISM_COUNT = 3   
TOURNAMENT_SIZE = 5     
COOLING_RATE = 0.995     
INITIAL_TEMPERATURE = 1000 

# **Early Stopping Parameters**
MAX_NO_IMPROVE = 10

class Individual:
    """
    Represents an individual in the population.
    Each individual has a route (list of city indices) and its corresponding fitness value.
    """
    def __init__(self, route: List[int]) -> None:
        self.route = route
        self.fitness = 0.0

    def calculate_fitness(self, distance_matrix: List[List[float]]) -> None:
        """
        Calculates the total distance of the route and assigns it as fitness.
        Lower fitness values are better.
        """
        total_distance = 0.0
        for i in range(len(self.route) - 1):
            from_city = self.route[i]
            to_city = self.route[i + 1]
            distance = distance_matrix[from_city][to_city]
            if distance == INT_MAX:
                self.fitness = INT_MAX
                return
            total_distance += distance
        self.fitness = total_distance

    # Define less-than for sorting individuals based on fitness
    def __lt__(self, other):
        return self.fitness < other.fitness

def read_tsp_file(file_path: str) -> Tuple[List[Tuple[float, float]], int]:
    """
    Reads the TSP file and extracts city coordinates and dimension.
    """
    nodes = {}
    coordinates = []
    dimension = 0
    reading_nodes = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())

            elif line.startswith('NODE_COORD_SECTION'):
                reading_nodes = True
                continue

            elif line == 'EOF':
                break

            if reading_nodes:
                if line:
                    parts = line.split()
                    if len(parts) >= 3:
                        node_id = int(parts[0]) - 1  # Adjusting to 0-based index
                        x = float(parts[1])
                        y = float(parts[2])
                        nodes[node_id] = (x, y)
                        coordinates.append((x, y))

    return coordinates, dimension

def compute_distance_matrix(coordinates: List[Tuple[float, float]], dimension: int) -> List[List[float]]:
    distance_matrix = [[0.0 for _ in range(dimension)] for _ in range(dimension)]

    for i in range(dimension):
        for j in range(i + 1, dimension):
            xi, yi = coordinates[i]
            xj, yj = coordinates[j]
            distance = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  

    return distance_matrix

def create_initial_population(dimension: int) -> List[Individual]:
    population = []
    base_route = list(range(dimension))
    for _ in range(POPULATION_SIZE):
        route = base_route.copy()
        random.shuffle(route[1:])  # Salesman always starts at city 0!
        route.append(route[0])     # Return to the starting city
        individual = Individual(route)
        population.append(individual)
    return population

def invert_subroute(route: List[int], start: int, end: int) -> List[int]:
    """
    Inverts the subroute between start and end indices.
    """
    new_route = route.copy()
    new_route[start:end] = reversed(new_route[start:end])
    return new_route

def mutate_route(route: List[int]) -> List[int]:
    """
    Applies inversion mutation to a route.
    """
    mutated_route = route.copy()
    dim = len(route) - 1  # Exclude the last city which is the starting point

    # Select two distinct positions for inversion
    start, end = sorted(random.sample(range(1, dim), 2))
    mutated_route = invert_subroute(mutated_route, start, end)
    return mutated_route

def partially_mapped_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Performs Partially Mapped Crossover (PMX) between two parents to produce an offspring.
    """
    size = len(parent1.route)
    child_route = [None] * size

    # Choose two random crossover points
    start, end = sorted(random.sample(range(1, size -1), 2))

    # Copy the subset from parent1 to child
    child_route[start:end] = parent1.route[start:end]

    # Mapping from parent2 to child
    for i in range(start, end):
        if parent2.route[i] not in child_route:
            gene = parent2.route[i]
            pos = i
            while True:
                gene_parent1 = parent1.route[pos]
                if gene_parent1 in parent2.route[start:end]:
                    pos = parent2.route.index(gene_parent1)
                else:
                    break
            child_route[pos] = gene

    # Fill the remaining positions with genes from parent2
    for i in range(1, size -1):
        if child_route[i] is None:
            child_route[i] = parent2.route[i]

    # Set the starting and ending city
    child_route[0] = parent1.route[0]
    child_route[-1] = parent1.route[0]

    return Individual(child_route)

def local_search_2opt(route: List[int], distance_matrix: List[List[float]]) -> List[int]:
    """
    Applies the 2-opt local search algorithm to improve the route.
    """
    best_route = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1: continue  # Skip adjacent edges
                new_route = route.copy()
                new_route[i:j] = reversed(new_route[i:j])
                # Calculate the difference in distance
                delta = (
                    distance_matrix[route[i - 1]][new_route[i]] +
                    distance_matrix[new_route[j - 1]][route[j]] -
                    distance_matrix[route[i - 1]][route[i]] -
                    distance_matrix[route[j - 1]][route[j]]
                )
                if delta < -1e-6:
                    best_route = new_route
                    route = new_route
                    improved = True
        if improved:
            break  # For increased computation expense, remove this break to allow full optimization
    return best_route

def crossover(parent1: Individual, parent2: Individual, distance_matrix: List[List[float]]) -> Individual:
    """
    Performs crossover between two parents and applies local search to the offspring.
    """
    child = partially_mapped_crossover(parent1, parent2)
    # Apply local search to the child to refine the route
    child.route = local_search_2opt(child.route, distance_matrix)
    return child

def calculate_fitness_wrapper(args):
    """
    Wrapper function for parallel fitness calculation.
    """
    individual, distance_matrix = args
    individual.calculate_fitness(distance_matrix)
    return individual

def calculate_population_fitness(population: List[Individual], distance_matrix: List[List[float]]) -> None:
    """
    Calculates and assigns fitness values for all individuals in the population using multiprocessing.
    """
    with multiprocessing.Pool() as pool:
        results = pool.map(calculate_fitness_wrapper, [(ind, distance_matrix) for ind in population])
    for i, individual in enumerate(results):
        population[i].fitness = individual.fitness

def selection(population: List[Individual]) -> List[Individual]:
    """
    Selects individuals to form a mating pool using a combination of tournament and rank selection.
    """
    mating_pool = []
    for _ in range(POPULATION_SIZE - ELITISM_COUNT):
        # Tournament selection
        tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament.sort()
        winner = tournament[0]
        mating_pool.append(winner)
    return mating_pool

def next_generation(current_gen: List[Individual], distance_matrix: List[List[float]], temperature: float) -> Tuple[List[Individual], float]:
    """
    Creates the next generation from the current generation.
    """
    new_population = []

    # Elitism: retain the top ELITISM_COUNT individuals
    elites = sorted(current_gen)[:ELITISM_COUNT]
    new_population.extend(deepcopy(elites))  # Deep copy to prevent reference issues

    # Selection
    mating_pool = selection(current_gen)

    # Crossover and generate offspring
    for i in range(0, len(mating_pool), 2):
        parent1 = mating_pool[i]
        parent2 = mating_pool[i + 1] if (i + 1) < len(mating_pool) else mating_pool[0]
        child1 = crossover(parent1, parent2, distance_matrix)
        child2 = crossover(parent2, parent1, distance_matrix)
        new_population.extend([child1, child2])

    # Mutation with adaptive rate
    for individual in new_population[ELITISM_COUNT:]:
        if random.random() < MUTATION_RATE:
            individual.route = mutate_route(individual.route)
            # Optionally apply local search after mutation
            individual.route = local_search_2opt(individual.route, distance_matrix)

    # Recalculate fitness for the entire new population
    calculate_population_fitness(new_population, distance_matrix)

    # Cooling
    new_temperature = temperature * COOLING_RATE

    return new_population, new_temperature

def genetic_algorithm(coordinates: List[Tuple[float, float]], 
                      population_size: int = POPULATION_SIZE, 
                      generations: int = GENERATIONS, 
                      mutation_rate: float = MUTATION_RATE, 
                      elitism_count: int = ELITISM_COUNT,
                      initial_temperature: float = INITIAL_TEMPERATURE) -> Individual:
    """
    Executes the genetic algorithm to solve the TSP.
    """
    # Update global constants based on parameters
    global POPULATION_SIZE, MUTATION_RATE, ELITISM_COUNT
    POPULATION_SIZE = population_size
    MUTATION_RATE = mutation_rate
    ELITISM_COUNT = elitism_count

    dimension = len(coordinates)
    distance_matrix = compute_distance_matrix(coordinates, dimension) 

    # Initialize population
    population = create_initial_population(dimension)
    calculate_population_fitness(population, distance_matrix)

    # Evolution loop
    temperature = initial_temperature
    best_fitness_over_time = []
    
    # **Initialize Early Stopping Variables**
    best_fitness = min(ind.fitness for ind in population)
    best_individual = min(population, key=lambda ind: ind.fitness)
    no_improve_count = 0

    # **Create 'results' Directory**
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    for generation in range(1, generations + 1):
        # Sort population based on fitness (ascending order)
        population.sort()
        current_best = population[0]
        current_best_fitness = current_best.fitness
        best_fitness_over_time.append(current_best_fitness)
        print(f"Generation {generation}: Best Fitness = {current_best_fitness:.2f}")

        # **Check for Improvement**
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1

        # **Early Stopping Condition**
        if no_improve_count >= MAX_NO_IMPROVE:
            print(f"No improvement in {MAX_NO_IMPROVE} consecutive generations. Terminating early.")
            break

        # **Save Current Best Route Plot**
        plot_filename = os.path.join(results_dir, f'generation_{generation:03d}.png')
        plot_route(current_best, coordinates, output_file=plot_filename)

        # Create next generation
        population, temperature = next_generation(population, distance_matrix, temperature)

        # Early termination if temperature is too low
        if temperature < 1:
            print("Temperature has cooled sufficiently. Terminating early.")
            break

    # Final best individual
    population.sort()
    final_best_individual = population[0]

    # Plot fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_over_time, color='green')
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Total Distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fitness_over_time.png'), dpi=300)
    plt.close()
    print(f"Fitness over generations plot saved as {os.path.join(results_dir, 'fitness_over_time.png')}")

    return final_best_individual

def plot_route(best_individual: Individual, coordinates: List[Tuple[float, float]], output_file: str = 'best_route.png') -> None:
    """
    Plots the best route found and saves it as a PNG file.
    """
    route = best_individual.route
    x = [coordinates[city][0] for city in route]
    y = [coordinates[city][1] for city in route]

    plt.figure(figsize=(12, 12))
    plt.plot(x, y, 'o-', color='blue', markersize=5, linewidth=1.5)
    plt.title('Best Route Found by Enhanced Genetic Algorithm')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Annotate each city with its index
    for idx, (xi, yi) in enumerate(coordinates):
        plt.text(xi, yi, str(idx), fontsize=9, ha='right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Route plot saved as {output_file}")

def print_solution(best_individual: Individual, coordinates: List[Tuple[float, float]]) -> None:
    """
    Prints the best route found by the genetic algorithm.
    """
    print("\nBest Route Found:")
    route_cities = best_individual.route
    city_names = [f"{city}" for city in route_cities]
    print(" -> ".join(city_names))
    print(f"Total Distance: {best_individual.fitness:.2f}")

def main():
    file_path = 'ch150.tsp'  

    coordinates, dimension = read_tsp_file(file_path) 

    best_solution = genetic_algorithm(
        coordinates=coordinates,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        elitism_count=ELITISM_COUNT,
        initial_temperature=INITIAL_TEMPERATURE
    )

    print_solution(best_solution, coordinates)
    plot_route(best_solution, coordinates, output_file='results/best_route.png')  # Plot and save the final route

if __name__ == "__main__":
    main()
