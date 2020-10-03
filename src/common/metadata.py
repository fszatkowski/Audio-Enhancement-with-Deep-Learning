import json
import os
import time
from argparse import ArgumentParser
from dataclasses import MISSING, dataclass
from datetime import datetime

import constants


@dataclass
class Metadata:
    # training settings
    epochs: int
    patience: int

    # dataset settings
    transformations: str

    model_dir: str
    train_files: int = 0
    test_files: int = constants.TEST_FILES
    val_files: int = constants.VAL_FILES
    current_epoch: int = 0

    # data settings
    batch_size: int = constants.BATCH_SIZE

    input_sr: int = constants.INPUT_SR
    target_sr: int = constants.TARGET_SR
    input_samples: int = constants.INPUT_SIZE
    target_samples: int = constants.TARGET_SIZE
    input_step_samples: int = constants.INPUT_STEP
    target_step_samples: int = constants.TARGET_STEP

    training_steps: int = 0
    save_every_n_steps: int = constants.SAVE_EVERY_N_STEPS

    training_finished: bool = False
    last_saved: str = str(datetime.now())
    timestamp: float = time.perf_counter()
    training_hours: float = 0
    random_seed: int = 123

    def save_to_json(self, name):
        settings_path = os.path.join(self.model_dir, name)
        if not os.path.exists(self.model_dir):
            os.makedirs(settings_path)
        timestamp = time.perf_counter()
        time_passed = timestamp - self.timestamp
        self.training_hours += time_passed / 3600
        self.timestamp = timestamp
        self.last_saved = str(datetime.now())
        with open(settings_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def from_args(cls) -> "Metadata":
        parser = ArgumentParser()
        for key, _ in cls.__dataclass_fields__.items():
            # if has default, add parser arg with default, else set required = True
            if cls.__dataclass_fields__[key].default != MISSING:
                parser.add_argument(
                    f"--{key}",
                    type=cls.__dataclass_fields__[key].type,
                    default=cls.__dataclass_fields__[key].default,
                )
            else:
                parser.add_argument(
                    f"--{key}", type=cls.__dataclass_fields__[key].type, required=True
                )
        args = parser.parse_args()
        metadata = cls(**args.__dict__)
        return metadata

    @staticmethod
    def restore_from_json(metadata: "Metadata", metadata_path: str) -> "Metadata":
        with open(metadata_path, "r") as f:
            data = json.load(f)
        epochs = metadata.epochs
        metadata.__dict__ = data
        metadata.epochs = epochs
        metadata.timestamp = time.perf_counter()

    @staticmethod
    def get_mock() -> "Metadata":
        return Metadata(
            epochs=1,
            patience=1,
            batch_size=1,
            test_files=1,
            train_files=1,
            val_files=1,
            transformations="none",
            input_sr=22050,
            target_sr=44100,
            input_samples=8192,
            target_samples=16384,
            input_step_samples=2048,
            target_step_samples=4096,
            model_dir="models/mock_model",
        )
