import os
from copy import deepcopy

import torch
from tqdm import tqdm

from common.dataloader import DataLoader
from common.dataset import Dataset
from common.evaluate import evaluate
from common.metadata import Metadata
from common.model_wrapper import ModelWrapper
from common.summary import TrainingSummary
from common.transformations_manager import TransformationsManager
from common.utils import ModelType, create_data_loaders
from constants import MODEL_FILENAME, TRAINING_RESULTS_FILENAME


def train_model(
    metadata: Metadata,
    wrapper: ModelWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    gan: bool = False,
):
    patience_ctr = 0

    print("\nStarting training.")

    training_summary = TrainingSummary.get(train_loader.batch_size, metadata, gan)
    for i, epoch in enumerate(range(metadata.current_epoch, metadata.epochs)):
        # train on whole train set
        train_loader.tm_active = (i >= metadata.warmup_epochs)
        for batch_ctr, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}.")):
            for mini_batch in batch:
                batch_loss = wrapper.train_step(mini_batch)

                current_batch_size = mini_batch[0].shape[0]
                training_summary.add_step(
                    batch_loss, current_batch_size=current_batch_size
                )

        # evaluate on val set
        val_loss = evaluate(wrapper, val_loader)

        training_summary.add_training_epoch(epoch)
        training_summary.add_val_loss(val_loss)
        training_summary.update_metadata(metadata)
        metadata.train_transformations = train_loader.tm.get_info()
        metadata.val_transformations = val_loader.tm.get_info()

        metadata.current_epoch = epoch + 1
        metadata.save_to_json(TRAINING_RESULTS_FILENAME)

        if training_summary.val_loss_improved():
            wrapper.save()
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr == metadata.patience:
            print(
                f"\nStopping training since validation loss hasn't improved for {metadata.patience} epochs."
            )
            break

    wrapper.load()
    print(f"\nTraining finished.")


def run_full_pipeline(
    metadata: Metadata, model_wrapper: ModelWrapper, model_type: ModelType
):
    print_cuda_info()
    restore_transformations = False

    if os.path.exists(os.path.join(metadata.model_dir, MODEL_FILENAME)):
        Metadata.restore_from_json(metadata, f"{model_wrapper.model_dir}/metadata.json")
        if metadata.training_finished:
            print(f"\n\n\nModel at {metadata.model_dir} already finished training.")
            return
        else:
            print(
                f"\n\n\nModel at {metadata.model_dir} already exists, restoring this model."
            )
            model_wrapper.load()
            restore_transformations = True
    else:
        os.makedirs(metadata.model_dir, exist_ok=True)

    metadata.num_params = model_wrapper.num_parameters()

    dataset = Dataset(metadata)

    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        metadata,
        model=model_type,
        transformations=TransformationsManager.get_transformations(
            metadata.transformations
        ),
    )

    if restore_transformations:
        train_loader.tm.transformations_count = metadata.train_transformations[
            "transformations_count"
        ]
        val_loader.tm.transformations_count = metadata.val_transformations[
            "transformations_count"
        ]

    train_model(
        metadata=metadata,
        wrapper=model_wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
        gan=(model_type == ModelType.SEGAN),
    )

    test_mse_loss = evaluate(model_wrapper, test_loader)
    print(f"Test set mse loss: {test_mse_loss}")

    metadata.test_mse_loss = test_mse_loss
    metadata.training_finished = True
    metadata.test_transformations = test_loader.tm.get_info()

    metadata.save_to_json(TRAINING_RESULTS_FILENAME)


def run_overfit(metadata: Metadata, model_wrapper: ModelWrapper, model_type: ModelType):
    print_cuda_info()

    os.makedirs(metadata.model_dir, exist_ok=True)

    if os.path.exists(os.path.join(metadata.model_dir, MODEL_FILENAME)):
        Metadata.restore_from_json(metadata, f"{model_wrapper.model_dir}/metadata.json")
        if metadata.training_finished:
            print(f"\n\n\nModel at {metadata.model_dir} already finished training.")
            return
        else:
            print(
                f"\n\n\nModel at {metadata.model_dir} already exists, restoring this model."
            )
            model_wrapper.load()
    else:
        os.makedirs(metadata.model_dir, exist_ok=True)

    metadata.num_params = model_wrapper.num_parameters()

    dataset = Dataset(metadata)

    train_loader, _, _ = create_data_loaders(
        dataset,
        metadata,
        model=model_type,
        transformations=TransformationsManager.get_transformations("none"),
    )

    eval_loader = deepcopy(train_loader)
    # if training gan, disable additional noisy inputs from datasets
    if model_type == ModelType.SEGAN:
        eval_loader.train_gan = False

    train_model(
        metadata=metadata,
        wrapper=model_wrapper,
        train_loader=train_loader,
        val_loader=eval_loader,
        gan=(model_type == ModelType.SEGAN),
    )

    test_mse_loss = evaluate(model_wrapper, eval_loader)
    print(f"Final mse loss: {test_mse_loss}")

    metadata.final_mse_loss = test_mse_loss
    metadata.training_finished = True
    metadata.save_to_json(TRAINING_RESULTS_FILENAME)


def print_cuda_info():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}.")
    else:
        print("Using CPU.")
