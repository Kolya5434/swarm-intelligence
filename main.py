import random
import numpy as np

coordinates = [(0, 0), (2, 4), (5, 5), (7, 3), (10, 2), (4, 8)]

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def fitness(solution):
    return sum(distance(coordinates[a], coordinates[b]) for a, b in zip(solution, solution[1:]))

def generate_population(size):
    population = []
    for _ in range(size):
        individual = list(range(len(coordinates)))
        random.shuffle(individual)
        population.append(individual)
    return population

def select(population, fitnesses, k=3):
    selected = random.choices(population, weights=fitnesses, k=k)
    return selected

def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for i in range(size):
        if child[i] is None:
            while parent2[pointer] in child:
                pointer += 1
            child[i] = parent2[pointer]
    return child

def mutate(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

def genetic_algorithm(population_size, generations):
    population = generate_population(population_size)
    for generation in range(generations):
        fitnesses = [1 / fitness(ind) for ind in population]
        new_population = []
        for _ in range(population_size // 2):
            parents = select(population, fitnesses, k=2)
            child1 = crossover(parents[0], parents[1])
            child2 = crossover(parents[1], parents[0])
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
        best_solution = min(population, key=fitness)
        print(f"Generation {generation}, Best solution fitness: {fitness(best_solution)}")
    return best_solution

best_route = genetic_algorithm(population_size=100, generations=500)
print("Найкраща знайдена послідовність точок:", best_route)
print("Мінімальна довжина трубопроводів:", fitness(best_route))
