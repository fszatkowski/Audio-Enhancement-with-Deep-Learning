from abc import ABC, abstractmethod
from typing import Tuple, Union

import numpy as np
import torch

Tensor = Union[torch.Tensor, np.array]


class Transformation(ABC):
    @abstractmethod
    def __init__(self, apply_probability: float):
        self.apply_probability = apply_probability

    @abstractmethod
    def apply(self, input_tensor: Tensor) -> Tensor:
        pass


class UniformNoiseFull(Transformation):
    def __init__(self, apply_probability: float, amplitude: float = 0.002):
        super(UniformNoiseFull, self).__init__(apply_probability)
        self.max = amplitude
        self.min = -amplitude

    def apply(self, input_tensor: Tensor) -> Tensor:
        return add_uniform_noise(
            input_tensor=input_tensor, min_val=self.min, max_val=self.max
        )


class GaussianNoiseFull(Transformation):
    def __init__(self, apply_probability: float, mean: float = 0.0, std: float = 0.001):
        super(GaussianNoiseFull, self).__init__(apply_probability)
        self.mean = mean
        self.std = std

    def apply(self, input_tensor: Tensor) -> Tensor:
        return add_gaussian_noise(
            input_tensor=input_tensor, mean=self.mean, std=self.std
        )


class UniformNoisePartial(Transformation):
    def __init__(
        self, apply_probability: float, noise_percent: float, amplitude: float = 0.002
    ):
        super(UniformNoisePartial, self).__init__(apply_probability)
        self.max = amplitude
        self.min = -amplitude
        self.noise_percent = noise_percent

    def apply(self, input_tensor: Tensor) -> Tensor:
        return add_uniform_noise(
            input_tensor=input_tensor,
            min_val=self.min,
            max_val=self.max,
            noise_percent=self.noise_percent,
        )


class GaussianNoisePartial(Transformation):
    def __init__(
        self,
        apply_probability: float,
        noise_percent: float,
        mean: float = 0.0,
        std: float = 0.001,
    ):
        super(GaussianNoisePartial, self).__init__(apply_probability)
        self.mean = mean
        self.std = std
        self.noise_percent = noise_percent

    def apply(self, input_tensor: Tensor) -> Tensor:
        return add_gaussian_noise(
            input_tensor=input_tensor,
            mean=self.mean,
            std=self.std,
            noise_percent=self.noise_percent,
        )


class ZeroSamplesTransformation(Transformation):
    def __init__(self, apply_probability: float, noise_percent: float):
        super(ZeroSamplesTransformation, self).__init__(apply_probability)
        self.noise_percent = noise_percent

    def apply(self, input_tensor: Tensor) -> Tensor:
        return set_value(
            input_tensor=input_tensor, percent_affected=self.noise_percent, value=0
        )


class ImpulseNoiseTransformation(Transformation):
    def __init__(
        self, apply_probability: float, impulse_value: float, noise_percent: float
    ):
        super(ImpulseNoiseTransformation, self).__init__(apply_probability)
        self.impulse_value = impulse_value
        self.noise_percent = noise_percent

    def apply(self, input_tensor: Tensor) -> Tensor:
        negative_amplitude = set_value(
            input_tensor=input_tensor,
            percent_affected=self.noise_percent / 2,
            value=-self.impulse_value,
        )
        return set_value(
            input_tensor=negative_amplitude,
            percent_affected=self.noise_percent / 2,
            value=self.impulse_value,
        )


def add_gaussian_noise(
    input_tensor: Tensor, mean: float, std: float, noise_percent: float = 1.0
) -> Tensor:
    noise = torch.empty(input_tensor.shape).normal_(mean=mean, std=std)
    noise = apply_random_mask(noise, noise_percent)
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor + noise
    elif isinstance(input_tensor, np.ndarray):
        return (torch.Tensor(input_tensor) + noise).numpy()


def add_uniform_noise(
    input_tensor: Tensor, min_val: float, max_val: float, noise_percent: float = 1.0
) -> Tensor:
    noise = (max_val - min_val) * torch.rand(input_tensor.shape) - min_val
    noise = apply_random_mask(noise, noise_percent)
    if isinstance(input_tensor, torch.Tensor):
        return input_tensor + noise
    elif isinstance(input_tensor, np.ndarray):
        return (torch.Tensor(input_tensor) + noise).numpy()


def apply_random_mask(noise: Tensor, percent: float) -> Tensor:
    if percent < 1.0:
        return noise * (torch.rand(noise.shape) <= percent).type(torch.int32)
    else:
        return noise


def set_value(input_tensor: Tensor, percent_affected: float, value: float) -> Tensor:
    convert_to_torch = isinstance(input_tensor, torch.Tensor)
    if convert_to_torch:
        input_tensor = input_tensor.numpy()
    input_tensor = input_tensor.copy()
    indices = np.random.choice(
        np.arange(input_tensor.size),
        replace=False,
        size=int(input_tensor.size * percent_affected),
    )
    indices = np.unravel_index(indices, input_tensor.shape)
    input_tensor[indices] = value
    if convert_to_torch:
        return torch.Tensor(input_tensor)
    return input_tensor
