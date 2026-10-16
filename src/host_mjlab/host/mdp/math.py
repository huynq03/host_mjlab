import torch
import numpy as np


def sigmoid(x, value_at_1):
    scale = np.sqrt(-2 * np.log(value_at_1))
    return torch.exp(-0.5 * (x * scale) ** 2)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    lower, upper = bounds
    assert lower < upper
    assert margin >= 0

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, sigmoid(d.double(), value_at_margin))

    return value
