from random import randint
from typing import Dict, Sequence

from common.transformations import *


class TransformationsManager:
    def __init__(self, transformations: Sequence[Transformation], max_transformations_applied: int = 1):
        self.transformations = transformations
        self.max_transformations_applied = max_transformations_applied

        self.transformations_count = {
            transformation.__class__.__name__: 0
            for transformation in self.transformations
        }
        self.transformations_count["NoTransformation"] = 0

    def apply_transformations(self, tensor: torch.Tensor) -> torch.Tensor:
        transformed_count = 0
        for t in self.transformations:
            if torch.rand(1) < t.apply_probability:
                tensor = t.apply(tensor)
                self.transformations_count[t.__class__.__name__] += 1
                transformed_count += 1
                if transformed_count == self.max_transformations_applied:
                    break
        if not transformed_count:
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
            return [
                GaussianNoisePartial(0.35, max_noise_percent=0.5, mean=0.0, max_std=0.1)
            ]
        elif transformations_type == "white_part":
            return [UniformNoisePartial(0.35, max_noise_percent=0.5, max_amplitude=0.1)]
        if transformations_type == "gaussian_uni":
            return [GaussianNoiseFull(0.35, mean=0.0, max_std=0.1)]
        elif transformations_type == "white_uni":
            return [UniformNoiseFull(0.35, max_amplitude=0.1)]
        elif transformations_type == "zero":
            return [ZeroSamplesTransformation(0.35, max_noise_percent=0.05)]
        elif transformations_type == "zero_002":
            return [ZeroSamplesTransformation(0.35, max_noise_percent=0.02)]
        elif transformations_type == "zero_001":
            return [ZeroSamplesTransformation(0.35, max_noise_percent=0.01)]
        elif transformations_type == "mix":
            return [
                GaussianNoisePartial(0.1, max_noise_percent=0.5, mean=0.0, max_std=0.1),
                GaussianNoiseFull(0.1, mean=0.0, max_std=0.1),
                UniformNoisePartial(0.1, max_noise_percent=0.5, max_amplitude=0.1),
                UniformNoiseFull(0.1, max_amplitude=0.1),
                ZeroSamplesTransformation(0.1, max_noise_percent=0.01),
            ]
        elif transformations_type == "mix_no_zero":
            return [
                GaussianNoisePartial(0.1, max_noise_percent=0.5, mean=0.0, max_std=0.1),
                GaussianNoiseFull(0.1, mean=0.0, max_std=0.1),
                UniformNoisePartial(0.1, max_noise_percent=0.5, max_amplitude=0.1),
                UniformNoiseFull(0.1, max_amplitude=0.1),
            ]
        elif transformations_type == "mix_weaker_zero":
            return [
                GaussianNoisePartial(0.1, max_noise_percent=0.5, mean=0.0, max_std=0.1),
                GaussianNoiseFull(0.1, mean=0.0, max_std=0.1),
                UniformNoisePartial(0.1, max_noise_percent=0.5, max_amplitude=0.1),
                UniformNoiseFull(0.1, max_amplitude=0.1),
                ZeroSamplesTransformation(0.1, max_noise_percent=0.001),
            ]
        elif transformations_type == "full":
            return []
        elif transformations_type == "default":
            return [
                GaussianNoisePartial(
                    0.25, max_noise_percent=1.0, mean=0.0, max_std=0.15
                ),
                GaussianNoiseFull(0.1, mean=0.0, max_std=0.15),
                UniformNoisePartial(0.25, max_noise_percent=1.0, max_amplitude=0.15),
                UniformNoiseFull(0.1, max_amplitude=0.15),
                ZeroSamplesTransformation(0.15, max_noise_percent=0.05),
                ImpulseNoiseTransformation(
                    0.15, max_noise_percent=0.05, max_impulse_value=0.15
                ),
            ]
        else:
            raise ValueError(
                f"Specified transformations config: {transformations_type} not supported."
            )

    @staticmethod
    def get_with_transformations(transformations_type: str) -> "TransformationsManager":
        return TransformationsManager(
            TransformationsManager.get_transformations(transformations_type)
        )
