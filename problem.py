from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Problem(ABC):
    name: str
    size: int
    optimal: int

    @abstractmethod
    def evaluate(self, cells: np.ndarray) -> float:
        pass

    @abstractmethod
    def init(self) -> np.ndarray:
        pass
