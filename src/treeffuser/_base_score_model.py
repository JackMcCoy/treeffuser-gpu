import abc
from typing import Optional, List
import numpy as np
from jaxtyping import Float, Int

# This is the new, clean base class definition
class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        t: Int[np.ndarray, "batch"],
    ):
        pass

    @abc.abstractmethod
    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: "DiffusionSDE",
        cat_idx: Optional[List[int]] = None,
    ):
        pass
