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
    def _apply(self, input_tensor: Tensor) -> Tensor:
        pass

    def apply(self, input_tensor: Tensor) -> Tensor:
        tensor = self._apply(input_tensor)

        if isinstance(tensor, torch.Tensor):
            return torch.clip(tensor, -1, 1)
        elif isinstance(tensor, np.ndarray):
            return np.clip(tensor, -1, 1)


class UniformNoiseFull(Transformation):
    def __init__(self, apply_probability: float, max_amplitude: float = 0.002):
        super(UniformNoiseFull, self).__init__(apply_probability)
        self.amplitude = torch.distributions.Uniform(0, max_amplitude)

    def _apply(self, input_tensor: Tensor) -> Tensor:
        amplitude = self.amplitude.sample()
        return add_uniform_noise(
            input_tensor=input_tensor, min_val=-amplitude, max_val=amplitude
        )


class GaussianNoiseFull(Transformation):
    def __init__(
        self, apply_probability: float, mean: float = 0.0, max_std: float = 0.001
    ):
        super(GaussianNoiseFull, self).__init__(apply_probability)
        self.mean = mean
        self.std = torch.distributions.Uniform(0, max_std)

    def _apply(self, input_tensor: Tensor) -> Tensor:
        return add_gaussian_noise(
            input_tensor=input_tensor, mean=self.mean, std=self.std.sample()
        )


class UniformNoisePartial(Transformation):
    def __init__(
        self,
        apply_probability: float,
        max_noise_percent: float,
        max_amplitude: float = 0.002,
    ):
        super(UniformNoisePartial, self).__init__(apply_probability)
        self.amplitude = torch.distributions.Uniform(0, max_amplitude)
        self.noise_percent = torch.distributions.Uniform(0, max_noise_percent)

    def _apply(self, input_tensor: Tensor) -> Tensor:
        amplitude = self.amplitude.sample()
        return add_uniform_noise(
            input_tensor=input_tensor,
            min_val=-amplitude,
            max_val=amplitude,
            noise_percent=self.noise_percent.sample(),
        )


class GaussianNoisePartial(Transformation):
    def __init__(
        self,
        apply_probability: float,
        max_noise_percent: float,
        mean: float = 0.0,
        max_std: float = 0.001,
    ):
        super(GaussianNoisePartial, self).__init__(apply_probability)
        self.mean = mean
        self.std = torch.distributions.Uniform(0, max_std)
        self.noise_percent = torch.distributions.Uniform(0, max_noise_percent)

    def _apply(self, input_tensor: Tensor) -> Tensor:
        return add_gaussian_noise(
            input_tensor=input_tensor,
            mean=self.mean,
            std=self.std.sample(),
            noise_percent=self.noise_percent.sample(),
        )


class ZeroSamplesTransformation(Transformation):
    def __init__(self, apply_probability: float, max_noise_percent: float):
        super(ZeroSamplesTransformation, self).__init__(apply_probability)
        self.noise_percent = torch.distributions.Uniform(0, max_noise_percent)

    def _apply(self, input_tensor: Tensor) -> Tensor:
        return set_value(
            input_tensor=input_tensor,
            percent_affected=self.noise_percent.sample(),
            value=0,
        )


class ImpulseNoiseTransformation(Transformation):
    def __init__(
        self,
        apply_probability: float,
        max_impulse_value: float,
        max_noise_percent: float,
    ):
        super(ImpulseNoiseTransformation, self).__init__(apply_probability)
        self.impulse_value = torch.distributions.Uniform(0, max_impulse_value)
        self.noise_percent = torch.distributions.Uniform(0, max_noise_percent)

    def _apply(self, input_tensor: Tensor) -> Tensor:
        impulse_value = self.impulse_value.sample()
        noise_percent = self.noise_percent.sample()
        negative_amplitude = set_value(
            input_tensor=input_tensor,
            percent_affected=noise_percent / 2,
            value=-impulse_value,
        )
        return set_value(
            input_tensor=negative_amplitude,
            percent_affected=noise_percent / 2,
            value=impulse_value,
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
