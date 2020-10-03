import numpy as np


def get_weights(name: str, array_size: int) -> np.array:
    weights = np.ones(array_size)
    if name == "uniform":
        pass
    elif name == "tri":
        indices = np.array(list([i for i in range(1, array_size + 1)]))
        quant = 1 / (array_size / 2)
        weights[: (array_size // 2)] = indices[: (array_size // 2)] * quant
        weights[(array_size // 2) :] = (
            array_size - (indices[(array_size // 2) :] - 1)
        ) * quant
    elif name == "tz_1":
        indices = np.array(list([i for i in range(1, array_size + 1)]))
        quant = 1 / ((array_size / 4) + 1)
        weights[: (array_size // 4)] = indices[: (array_size // 4)] * quant
        weights[3 * (array_size // 4) :] = (
            array_size - (indices[3 * (array_size // 4) :] - 1)
        ) * quant
    elif name == "tz_2":
        indices = np.array(list([i for i in range(1, array_size + 1)]))
        quant = 0.5 / (array_size / 4)
        weights[: (array_size // 4)] = (
            0.5 + indices[: (array_size // 4)] * quant - quant
        )
        weights[3 * (array_size // 4) :] = (
            0.5 + (array_size - (indices[3 * (array_size // 4) :] - 1)) * quant - quant
        )
    else:
        raise ValueError(f"Unknown weights requested: {name}")

    return weights
