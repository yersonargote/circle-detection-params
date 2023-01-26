from dataclasses import dataclass

import numpy as np

from circle import CircleDetection
from gwo import GWO
from problem import Problem
from solution import Solution

# from time import perf_counter




@dataclass
class Params(Problem):
    edges: np.ndarray

    def init(self) -> np.ndarray:
        N = np.random.randint(3, 200)
        max_iterations = np.random.randint(1, 1000)

        return np.array([N, max_iterations])

    def evaluate(self, cells: np.ndarray) -> float:
        N, max_iterations = np.int32(cells)
        if N < 3:
            N = 3
        if max_iterations < 1:
            max_iterations = 1
        name = "Circle Detection"
        size, optimal = (3, 0)
        min_radius, max_radius = (50, 200)
        problem: CircleDetection = CircleDetection(
            name=name,
            size=size,
            min_radius=min_radius,
            max_radius=max_radius,
            optimal=optimal,
            edges=self.edges,
        )

        gwo: GWO = GWO(
            max_iterations=max_iterations,
            N=N,
            problem=problem,
            population=np.empty(shape=N, dtype=object),
            a=0,
            alpha=Solution(np.zeros(size), np.Inf),
            beta=Solution(np.zeros(size), np.Inf),
            delta=Solution(np.zeros(size), np.Inf),
        )
        # start = perf_counter()
        best = gwo.solve()
        # end = perf_counter()
        # time = end - start
        return best.fitness
