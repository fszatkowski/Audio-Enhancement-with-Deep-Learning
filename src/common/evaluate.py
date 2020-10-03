import numpy as np
import torch
import torch.utils.data as data

from common.model_wrapper import ModelWrapper


def evaluate(wrapper: ModelWrapper, loader: data.DataLoader) -> float:
    wrapper.net.eval()
    with torch.no_grad():
        ctr: float = 0
        cumulative_mse: float = 0

        for batch in loader:
            for mini_batch in batch:
                cumulative_mse += wrapper.compute_mse_loss(*mini_batch).item()
                ctr += mini_batch[0].shape[0] / loader.batch_size
        wrapper.net.train()
        return float(np.log10(cumulative_mse / ctr))
