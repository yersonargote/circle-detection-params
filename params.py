from dataclasses import dataclass
from time import perf_counter

import numpy as np

from circle import CircleDetection
from gwo import GWO
from problem import Problem
from solution import Solution


@dataclass
class Params(Problem):
    edges: np.ndarray

    def init(self) -> np.ndarray:
        N = np.random.randint(3, 80)
        max_iterations = np.random.randint(1, 400)

        return np.array([N, max_iterations])

    def evaluate(self, cells: np.ndarray) -> float:
        N, max_iterations = np.int16(cells)
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
        start = perf_counter()
        best = gwo.solve()
        end = perf_counter()
        time = end - start
        return 0.9 * best.fitness + 0.1 * time
