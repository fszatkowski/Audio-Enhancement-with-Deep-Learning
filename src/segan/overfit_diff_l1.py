import constants
from segan.metadata import SEGANMetadata
from segan.overfit import train

if __name__ == "__main__":
    for l1 in [0.001, 0.01, 0.1, 1, 10, 100, 100]:
        train(
            SEGANMetadata(
                epochs=constants.OVERFIT_EPOCHS,
                patience=constants.OVERFIT_PATIENCE,
                batch_size=constants.OVERFIT_BATCH_SIZE,
                train_files=32,
                val_files=1,
                test_files=1,
                save_every_n_steps=constants.OVERFIT_N_STEPS,
                transformations="none",
                model_dir=f"models/segan/overfit_model_l1_{l1}",
                l1_alpha=l1,
            )
        )
