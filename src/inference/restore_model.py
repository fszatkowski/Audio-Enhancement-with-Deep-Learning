import os
from typing import Any, Dict

import torch

from autoencoder.modules import AutoencoderModule
from constants import MODEL_FILENAME
from segan.main_module import SEGANModule
from wavenet.modules import WaveNetModule


def restore_model(metadata: Dict[str, Any]) -> torch.nn.Module:
    model_path = os.path.join(metadata["model_dir"], MODEL_FILENAME)
    model_type = os.path.dirname(model_path).split("/")[-2]

    model: torch.nn.Module
    if model_type == "autoencoder":
        model = AutoencoderModule(
            num_layers=metadata["num_layers"],
            channels=metadata["channels"],
            kernel_size=metadata["kernel_size"],
            norm=metadata["norm"],
            activation=metadata["activation"],
        )
    elif model_type == "wavenet":
        model = WaveNetModule(
            blocks_per_stack=metadata["stack_layers"],
            stack_size=metadata["stack_size"],
            input_kernel_size=metadata["input_kernel_size"],
            res_channels=metadata["residual_channels"],
            skip_kernel_size=metadata["skip_kernel_size"],
            skip_channels=metadata["skip_channels"],
        )
    elif model_type == "segan":
        model = SEGANModule(
            n_layers=metadata["n_layers"],
            init_channels=metadata["init_channels"],
            kernel_size=metadata["kernel_size"],
            stride=metadata["stack_layers"],
            d_linear_units=metadata["input_samples"]
            // (metadata["multiplier"] ** metadata["n_layers"]),
            g_norm=metadata["g_norm"],
            d_norm=metadata["d_norm"],
        )
    else:
        raise ValueError(f"Model type unknown: {model_type}")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    if model_type == "segan":
        model = model.generator
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model
