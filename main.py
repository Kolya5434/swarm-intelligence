import numpy as np
import random

class Particle:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.position = np.random.permutation(num_nodes)  # Random path
        self.best_position = self.position.copy()
        self.best_cost = float('inf')  # Initialize with infinity
        self.velocity = np.zeros(num_nodes)  # Velocity is initially zero

    def evaluate(self, cost_matrix):
        # Calculate the cost of the current path
        cost = 0
        for i in range(len(self.position) - 1):
            cost += cost_matrix[self.position[i], self.position[i + 1]]
        return cost

    def update_velocity(self, global_best_position):
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive (particle) weight
        c2 = 1.5  # Social (global) weight

        for i in range(self.num_nodes):
            r1 = random.random()
            r2 = random.random()
            if r1 < 0.5:  # Randomly decide whether to swap two nodes
                j = random.randint(0, self.num_nodes - 1)
                self.position[i], self.position[j] = self.position[j], self.position[i]

    def update_position(self):
        self.position = np.unique(self.position)  # Ensure the path is unique

class PSO:
    def __init__(self, cost_matrix, num_particles, max_iterations):
        self.cost_matrix = cost_matrix
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.particles = [Particle(len(cost_matrix)) for _ in range(num_particles)]
        self.global_best_cost = float('inf')
        self.global_best_position = None

    def optimize(self):
        for _ in range(self.max_iterations):
            for particle in self.particles:
                cost = particle.evaluate(self.cost_matrix)
                if cost < particle.best_cost:
                    particle.best_cost = cost
                    particle.best_position = particle.position.copy()

                if cost < self.global_best_cost:
                    self.global_best_cost = cost
                    self.global_best_position = particle.position.copy()

            for particle in self.particles:
                particle.update_velocity(self.global_best_position)

        return self.global_best_position, self.global_best_cost

def main():
    cost_matrix = np.array([[0, 2, 9, 10],
                             [1, 0, 6, 4],
                             [15, 7, 0, 8],
                             [6, 3, 12, 0]])

    # Print the cost matrix
    print("Cost matrix:")
    print(cost_matrix)

    num_particles = 30
    max_iterations = 100

    pso = PSO(cost_matrix, num_particles, max_iterations)
    best_path, best_cost = pso.optimize()

    print("Best path:", best_path)
    print("Best cost:", best_cost)

if __name__ == "__main__":
    main()