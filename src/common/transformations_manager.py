from random import randint
from typing import Dict, Sequence

import torch

from common.transformations import (GaussianNoisePartial, GaussianNoiseUniform,
                                    Transformation, WhiteNoisePartial,
                                    WhiteNoiseUniform,
                                    ZeroSamplesTransformation)


class TransformationsManager:
    def __init__(self, transformations: Sequence[Transformation]):
        self.transformations = transformations
        self.transformations_count = {
            transformation.__class__.__name__: 0
            for transformation in self.transformations
        }
        self.transformations_count["NoTransformation"] = 0

    def apply_transformations(self, tensor: torch.Tensor) -> torch.Tensor:
        transformed = False
        for t in self.transformations:
            if torch.rand(1) < t.apply_probability:
                tensor = t.apply(tensor)
                self.transformations_count[t.__class__.__name__] += 1
                transformed = True
        if not transformed:
            self.transformations_count["NoTransformation"] += 1
        return tensor

    def apply_random_transformation(self, tensor: torch.Tensor) -> torch.Tensor:
        random_idx = randint(0, len(self.transformations) - 1)
        return self.transformations[random_idx].apply(tensor)

    def clear_history(self):
        self.transformations_count = {
            t.__class__.__name__: 0 for t in self.transformations
        }
        self.transformations_count["NoTransformation"] = 0

    def get_info(self) -> Dict[str, Dict]:
        return {
            "transformations": {
                transformation.__class__.__name__: transformation.__dict__
                for transformation in self.transformations
            },
            "transformations_count": self.transformations_count,
        }

    @staticmethod
    def get_transformations(transformations_type: str) -> Sequence[Transformation]:
        if transformations_type == "gaussian_part":
            return [GaussianNoisePartial(0.35, noise_percent=0.5, mean=0.0, std=0.1)]
        elif transformations_type == "white_part":
            return [WhiteNoisePartial(0.35, noise_percent=0.5, amplitude=0.1)]
        if transformations_type == "gaussian_uni":
            return [GaussianNoiseUniform(0.35, mean=0.0, std=0.1)]
        elif transformations_type == "white_uni":
            return [WhiteNoiseUniform(0.35, amplitude=0.1)]
        elif transformations_type == "zero":
            return [ZeroSamplesTransformation(0.35, noise_percent=0.05)]
        elif transformations_type == "zero_002":
            return [ZeroSamplesTransformation(0.35, noise_percent=0.02)]
        elif transformations_type == "zero_001":
            return [ZeroSamplesTransformation(0.35, noise_percent=0.01)]
        elif transformations_type == "mix":
            return [
                GaussianNoisePartial(0.1, noise_percent=0.5, mean=0.0, std=0.1),
                GaussianNoiseUniform(0.1, mean=0.0, std=0.1),
                WhiteNoisePartial(0.1, noise_percent=0.5, amplitude=0.1),
                WhiteNoiseUniform(0.1, amplitude=0.1),
                ZeroSamplesTransformation(0.1, noise_percent=0.01),
            ]
        elif transformations_type == "mix_no_zero":
            return [
                GaussianNoisePartial(0.1, noise_percent=0.5, mean=0.0, std=0.1),
                GaussianNoiseUniform(0.1, mean=0.0, std=0.1),
                WhiteNoisePartial(0.1, noise_percent=0.5, amplitude=0.1),
                WhiteNoiseUniform(0.1, amplitude=0.1),
            ]
        elif transformations_type == "mix_weaker_zero":
            return [
                GaussianNoisePartial(0.1, noise_percent=0.5, mean=0.0, std=0.1),
                GaussianNoiseUniform(0.1, mean=0.0, std=0.1),
                WhiteNoisePartial(0.1, noise_percent=0.5, amplitude=0.1),
                WhiteNoiseUniform(0.1, amplitude=0.1),
                ZeroSamplesTransformation(0.1, noise_percent=0.001),
            ]
        elif transformations_type == "none":
            return []
        else:
            raise ValueError(
                f"Specified transformations config: {transformations_type} not supported."
            )

    @staticmethod
    def get_with_transformations(transformations_type: str) -> "TransformationsManager":
        return TransformationsManager(
            TransformationsManager.get_transformations(transformations_type)
        )
