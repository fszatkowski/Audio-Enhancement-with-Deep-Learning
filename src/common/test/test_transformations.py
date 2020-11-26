import numpy as np
import torch

from common.transformations import *


def test_numpy():
    array = np.zeros((10, 20, 30))

    transform = GaussianNoiseFull(apply_probability=1, mean=0, max_std=0.2)
    transformed_array = transform.apply(array)
    assert np.all(0 != transformed_array)
    assert np.all(array != transformed_array)
    assert np.all(array == 0)

    transform = GaussianNoisePartial(
        apply_probability=1, mean=0, max_std=0.2, max_noise_percent=0.5
    )
    transformed_array = transform.apply(array)
    assert np.any(0 == transformed_array)
    assert np.all(array == 0)

    transform = UniformNoiseFull(apply_probability=1, max_amplitude=1)
    transformed_array = transform.apply(array)
    assert np.all(0 != transformed_array)
    assert np.all(array != transformed_array)
    assert np.all(array == 0)

    transform = UniformNoisePartial(
        apply_probability=1, max_amplitude=1, max_noise_percent=0.5
    )
    transformed_array = transform.apply(array)
    assert np.any(0 == transformed_array)
    assert np.all(array == 0)

    transform = ImpulseNoiseTransformation(
        apply_probability=1, max_impulse_value=1, max_noise_percent=0.5
    )
    transformed_array = transform.apply(array)
    assert np.any(0.0 != transformed_array)
    assert np.all(array == 0)

    array = array + 1
    transform = ZeroSamplesTransformation(apply_probability=1, max_noise_percent=0.5)
    transformed_array = transform.apply(array)
    assert np.any(0 == transformed_array)
    assert np.all(array == 1)


def test_torch():
    array = torch.zeros((10, 20, 30))

    transform = GaussianNoiseFull(apply_probability=1, mean=0, max_std=0.2)
    transformed_array = transform.apply(array)
    assert torch.all(0 != transformed_array)
    assert torch.all(array != transformed_array)
    assert torch.all(array == 0)

    transform = GaussianNoisePartial(
        apply_probability=1, mean=0, max_std=0.2, max_noise_percent=0.5
    )
    transformed_array = transform.apply(array)
    assert torch.any(0 == transformed_array)
    assert torch.all(array == 0)

    transform = UniformNoiseFull(apply_probability=1, max_amplitude=1)
    transformed_array = transform.apply(array)
    assert torch.all(0 != transformed_array)
    assert torch.all(array != transformed_array)
    assert torch.all(array == 0)

    transform = UniformNoisePartial(
        apply_probability=1, max_amplitude=1, max_noise_percent=0.5
    )
    transformed_array = transform.apply(array)
    assert torch.any(0 == transformed_array)
    assert torch.all(array == 0)

    transform = ImpulseNoiseTransformation(
        apply_probability=1, max_impulse_value=1, max_noise_percent=0.5
    )
    transformed_array = transform.apply(array)
    assert torch.any(0.0 != transformed_array)
    assert torch.all(array == 0)

    array = array + 1
    transform = ZeroSamplesTransformation(apply_probability=1, max_noise_percent=0.5)
    transformed_array = transform.apply(array)
    assert torch.any(0 == transformed_array)
    assert torch.all(array == 1)


if __name__ == "__main__":
    test_numpy()
    test_torch()
