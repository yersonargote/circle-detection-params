# -*- coding: utf-8 -*-
"""GWO-CircleDetection.ipynb"""

import cv2 as cv
import numpy as np
from rich import print

from ga import GA
# from gwo import GWO
from params import Params
from solution import Solution


def canny(filename: str):
    img = cv.imread(filename, 0)
    # edges = cv.Canny(img, 50, 50)
    edges = cv.Canny(img, 100, 200)
    return edges


def main():
    np.random.seed(42)

    max_iterations = 10
    N = 20
    size = 2
    optimal = 0

    edges = canny("2.jpg")

    problem: Params = Params(
        name="Params",
        optimal=optimal,
        size=size,
        edges=edges,
    )

    # gwo: GWO = GWO(
    #     max_iterations=max_iterations,
    #     N=N,
    #     problem=problem,
    #     population=np.empty(shape=N, dtype=object),
    #     a=0,
    #     alpha=Solution(np.zeros(size), np.Inf),
    #     beta=Solution(np.zeros(size), np.Inf),
    #     delta=Solution(np.zeros(size), np.Inf),
    # )
    # best = gwo.solve()
    # print(f"{best}")
    # print(gwo.population)

    ga = GA(
        N=N,
        generations=max_iterations,
        problem=problem,
        population=np.empty(shape=N, dtype=object),
        opponents=2,
    )
    best = ga.solve()
    print(f"{best}")
    print(ga.population)


if __name__ == "__main__":
    main()
