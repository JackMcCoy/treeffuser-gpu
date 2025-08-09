import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional, List
from jaxtyping import Float

# This file needs DiffusionSDE for type hints and runtime access.
from treeffuser.sde import DiffusionSDE

def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int,
    eval_percent: Optional[float],
    cat_idx: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """Creates the training data for the score model."""
    EPS = 1e-5
    T = sde.T
    if seed is not None:
        np.random.seed(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    if eval_percent is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_percent, random_state=seed
        )

    # TRAINING DATA
    X_train_tiled = np.tile(X_train, (n_repeats, 1))
    y_train_tiled = np.tile(y_train, (n_repeats, 1))
    t_train = np.random.uniform(0, 1, size=(y_train_tiled.shape[0], 1)) * (T - EPS) + EPS
    z_train = np.random.normal(size=y_train_tiled.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train_tiled, t_train)
    perturbed_y_train = train_mean + train_std * z_train
    predictors_train = np.concatenate([perturbed_y_train, X_train_tiled, t_train], axis=1)
    predicted_train = -1.0 * z_train

    # VALIDATION DATA
    predictors_val, predicted_val = None, None
    if eval_percent is not None:
        t_val = np.random.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = np.random.normal(size=(y_test.shape[0], y_test.shape[1]))

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = np.concatenate(
            [perturbed_y_val, X_test, t_val.reshape(-1, 1)], axis=1
        )
        predicted_val = -1.0 * z_val

    new_cat_idx = [c + y_train.shape[1] for c in cat_idx] if cat_idx is not None else None

    return predictors_train, predictors_val, predicted_train, predicted_val, new_cat_idx
