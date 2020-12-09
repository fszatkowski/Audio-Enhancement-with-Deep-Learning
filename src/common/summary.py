from datetime import datetime
from typing import List, Tuple, Union

import numpy as np

import constants
from common.metadata import Metadata


class TrainingSummary:
    def __init__(self, batch_size: int, metadata: Metadata):
        self.batch_size = batch_size
        self.warmup_epochs = metadata.warmup_epochs
        self.current_epoch = metadata.current_epoch

        if hasattr(metadata, "val_losses"):
            self.val_losses: List[float] = metadata.val_losses
        else:
            self.val_losses: List[float] = []
        self.ctr: float = 0

        self.steps: int = 0
        self.n_steps = metadata.save_every_n_steps

    @staticmethod
    def get(batch_size: int, metadata: Metadata, gan: bool) -> "TrainingSummary":
        if gan:
            return GANSummary(batch_size, metadata)

        return StandardSummary(batch_size, metadata)

    def _add_step(
        self,
        batch_loss: Union[float, Tuple[float, float, float, float]],
        current_batch_size: int,
    ):
        pass

    def add_step(
        self,
        batch_loss: Union[float, Tuple[float, float, float, float]],
        current_batch_size: int,
    ):
        self._add_step(batch_loss, current_batch_size)
        self.ctr += current_batch_size / self.batch_size

    def _add_training_epoch(self, epoch: int):
        pass

    def add_training_epoch(self, epoch: int):
        self._add_training_epoch(epoch)
        self.ctr = 0

    def add_epoch_val_loss(self, val_loss: float):
        self.val_losses.append(val_loss)
        print(f"{str(datetime.now())}: Validation loss: {val_loss}\n")
        self.current_epoch = self.current_epoch + 1

    def val_loss_improved(self) -> bool:
        if len(self.val_losses) <= self.warmup_epochs:
            return True
        else:
            return self.val_losses[-1] <= min(self.val_losses[self.warmup_epochs-1:])

    def _update_metadata(self, metadata: Metadata):
        pass

    def update_metadata(self, metadata: Metadata):
        self._update_metadata(metadata)

        metadata.training_steps = self.steps
        metadata.val_losses = self.val_losses
        metadata.final_val_loss = min(self.val_losses)


class StandardSummary(TrainingSummary):
    def __init__(self, batch_size: int, metadata: Metadata):
        super(StandardSummary, self).__init__(batch_size, metadata)

        if hasattr(metadata, "train_losses") and hasattr(
            metadata, "intermediate_train_losses"
        ):
            self.train_losses = metadata.train_losses
            self.intermediate_train_losses = metadata.intermediate_train_losses
        else:
            self.train_losses: List[float] = []
            self.intermediate_train_losses: List[float] = []

        self.cumulative_mse_loss: float = 0

    def _add_step(self, batch_loss: float, current_batch_size: int):
        self.cumulative_mse_loss += batch_loss
        self.steps += 1

        if self.steps % self.n_steps == 0:
            self.intermediate_train_losses.append(batch_loss)

    def _add_training_epoch(self, epoch: int):
        train_loss = np.log10(self.cumulative_mse_loss / self.ctr)
        self.train_losses.append(train_loss)
        print(
            f"{str(datetime.now())}: Epoch: {epoch} finished\n"
            f"training loss: {train_loss}"
        )

        self.cumulative_mse_loss = 0

    def _update_metadata(self, metadata: Metadata):
        metadata.train_losses = self.train_losses
        metadata.intermediate_train_losses = self.intermediate_train_losses

        metadata.final_train_loss = min(self.train_losses)


class GANSummary(TrainingSummary):
    def __init__(self, batch_size: int, metadata: Metadata):
        super(GANSummary, self).__init__(batch_size, metadata)

        if (
            hasattr(metadata, "train_d_real_losses")
            and hasattr(metadata, "train_d_fake_losses")
            and hasattr(metadata, "train_g_adv_losses")
            and hasattr(metadata, "train_g_l1_losses")
            and hasattr(metadata, "intermediate_train_d_real_losses")
            and hasattr(metadata, "intermediate_train_d_fake_losses")
            and hasattr(metadata, "intermediate_train_g_adv_losses")
            and hasattr(metadata, "intermediate_train_g_l1_losses")
        ):
            self.train_d_real_losses = metadata.train_d_real_losses
            self.train_d_fake_losses = metadata.train_d_fake_losses
            self.train_g_adv_losses = metadata.train_g_adv_losses
            self.train_g_l1_losses = metadata.train_g_l1_losses

            self.intermediate_train_d_real_losses = (
                metadata.intermediate_train_d_real_losses
            )
            self.intermediate_train_d_fake_losses = (
                metadata.intermediate_train_d_fake_losses
            )
            self.intermediate_train_g_adv_losses = (
                metadata.intermediate_train_g_adv_losses
            )
            self.intermediate_train_g_l1_losses = (
                metadata.intermediate_train_g_l1_losses
            )

        else:
            self.train_d_real_losses: List[float] = []
            self.train_d_fake_losses: List[float] = []
            self.train_g_adv_losses: List[float] = []
            self.train_g_l1_losses: List[float] = []

            self.intermediate_train_d_real_losses: List[float] = []
            self.intermediate_train_d_fake_losses: List[float] = []
            self.intermediate_train_g_adv_losses: List[float] = []
            self.intermediate_train_g_l1_losses: List[float] = []

        self.cumulative_d_real_loss: float = 0
        self.cumulative_d_fake_loss: float = 0
        self.cumulative_g_adv_loss: float = 0
        self.cumulative_g_l1_loss: float = 0

    def _add_step(
        self, batch_loss: Tuple[float, float, float, float], current_batch_size: int
    ):
        self.cumulative_d_real_loss += batch_loss[0]
        self.cumulative_d_fake_loss += batch_loss[1]
        self.cumulative_g_adv_loss += batch_loss[2]
        self.cumulative_g_l1_loss += batch_loss[3]
        self.steps += 1

        if self.steps % self.n_steps == 0:
            self.intermediate_train_d_real_losses.append(batch_loss[0])
            self.intermediate_train_d_fake_losses.append(batch_loss[1])
            self.intermediate_train_g_adv_losses.append(batch_loss[2])
            self.intermediate_train_g_l1_losses.append(batch_loss[3])

    def _add_training_epoch(self, epoch: int):
        train_d_real_loss = np.log10(self.cumulative_d_real_loss / self.ctr)
        self.train_d_real_losses.append(train_d_real_loss)
        train_d_fake_loss = np.log10(self.cumulative_d_fake_loss / self.ctr)
        self.train_d_fake_losses.append(train_d_fake_loss)
        train_g_adv_loss = np.log10(self.cumulative_g_adv_loss / self.ctr)
        self.train_g_adv_losses.append(train_g_adv_loss)
        train_g_l1_loss = np.log10(self.cumulative_g_l1_loss / self.ctr)
        self.train_g_l1_losses.append(train_g_l1_loss)
        print(
            f"{str(datetime.now())}: Epoch: {epoch} finished\n"
            f"discriminator real loss: {train_d_real_loss}\n"
            f"discriminator fake loss: {train_d_fake_loss}\n"
            f"generator adversarial loss: {train_g_adv_loss}\n"
            f"generator l1 loss: {train_g_l1_loss}"
        )

        self.cumulative_d_real_loss = 0
        self.cumulative_d_fake_loss = 0
        self.cumulative_g_adv_loss = 0
        self.cumulative_g_l1_loss = 0

    def _update_metadata(self, metadata: Metadata):
        metadata.train_d_real_loss = self.train_d_real_losses
        metadata.train_d_fake_loss = self.train_d_fake_losses
        metadata.train_g_adv_loss = self.train_g_adv_losses
        metadata.train_g_l1_loss = self.train_g_l1_losses

        metadata.intermediate_train_d_real_losses = (
            self.intermediate_train_d_real_losses
        )
        metadata.intermediate_train_d_fake_losses = (
            self.intermediate_train_d_fake_losses
        )
        metadata.intermediate_train_g_adv_losses = self.intermediate_train_g_adv_losses
        metadata.intermediate_train_g_l1_losses = self.intermediate_train_g_l1_losses
