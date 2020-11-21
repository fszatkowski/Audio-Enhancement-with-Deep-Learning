from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Transformation(ABC):
    @abstractmethod
    def __init__(self, apply_probability: float):
        self.apply_probability = apply_probability

    @abstractmethod
    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass


class UniformNoiseFull(Transformation):
    def __init__(self, apply_probability: float, amplitude: float = 0.002):
        super(UniformNoiseFull, self).__init__(apply_probability)
        self.max = amplitude
        self.min = -amplitude

    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        noise = uniform(input_tensor.shape, self.min, self.max)
        return input_tensor + noise


class GaussianNoiseFull(Transformation):
    def __init__(self, apply_probability: float, mean: float = 0.0, std: float = 0.001):
        super(GaussianNoiseFull, self).__init__(apply_probability)
        self.mean = mean
        self.std = std

    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        noise = normal(input_tensor.shape, self.mean, self.std)
        return input_tensor + noise


class UniformNoisePartial(Transformation):
    def __init__(
        self, apply_probability: float, noise_percent: float, amplitude: float = 0.002
    ):
        super(UniformNoisePartial, self).__init__(apply_probability)
        self.max = amplitude
        self.min = -amplitude
        self.noise_percent = noise_percent

    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        noise = uniform(input_tensor.shape, self.min, self.max) * random_mask(
            input_tensor.shape, self.noise_percent
        )
        return input_tensor + noise


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

    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        noise = normal(input_tensor.shape, self.mean, self.std) * random_mask(
            input_tensor.shape, self.noise_percent
        )
        return input_tensor + noise


class ZeroSamplesTransformation(Transformation):
    def __init__(self, apply_probability: float, noise_percent: float):
        super(ZeroSamplesTransformation, self).__init__(apply_probability)
        self.noise_percent = noise_percent

    def apply(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor * random_mask(input_tensor.shape, self.noise_percent)


def normal(shape: Tuple[int], mean: float, std: float) -> torch.Tensor:
    return torch.empty(shape).normal_(mean=mean, std=std)


def uniform(shape: Tuple[int], min_val: float, max_val: float) -> torch.Tensor:
    return (max_val - min_val) * torch.rand(shape) - min_val


def random_mask(shape: Tuple[int], percent: float) -> torch.Tensor:
    return (torch.rand(shape) <= percent).type(torch.int32)
