import random
import numpy as np

class FireflyOptimizer:
    def __init__(self, bounds, n_fireflies=20, alpha=0.2, beta=1.0, gamma=1.0):
        self.bounds = bounds
        self.n = n_fireflies
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = self._init_population()

    def _init_population(self):
        return [
            np.array([random.uniform(b[0], b[1]) for b in self.bounds])
            for _ in range(self.n)
        ]

    def optimize(self, fitness_fn, iters=20):
        for _ in range(iters):
            for i in range(self.n):
                for j in range(self.n):
                    if fitness_fn(self.population[j]) < fitness_fn(self.population[i]):
                        r = np.linalg.norm(self.population[i] - self.population[j])
                        self.population[i] += (
                            self.beta * np.exp(-self.gamma * r**2)
                            * (self.population[j] - self.population[i])
                            + self.alpha * np.random.randn(len(self.bounds))
                        )
        return min(self.population, key=fitness_fn)
