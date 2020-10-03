from os.path import join

FMA_PATH = "data/fma"
MP3_GLOB = join(FMA_PATH, "*/*/*.mp3")
PREPROCESSED_DATASET_DIR = join(FMA_PATH, "preprocessed_clean")
RAW_DATASET_FILELIST = "data/filelist.txt"
NUM_WORKERS = 0

OVERFIT_EPOCHS = 50
OVERFIT_PATIENCE = 10
OVERFIT_BATCH_SIZE = 4
OVERFIT_N_STEPS = 100

# shared across all models
INPUT_SR = 22050
TARGET_SR = 44100
INPUT_SIZE = 16384
TARGET_SIZE = 32768
INPUT_STEP = 4096
TARGET_STEP = 8192
BATCH_SIZE = 4

SAVE_EVERY_N_STEPS = 10000

OUTPUT_MODELS_DIR: str = "models/"
RESULTS_DIR: str = "results/"
OUTPUT_CSV = join(RESULTS_DIR, "stats.csv")
CSV_KEYS = [
    "model_dir",
    "final_val_loss",
    "test_mse_loss",
    "epochs",
    "training_hours",
    "batch_size",
    "num_params",
    "train_files",
    "patience",
    "transformations",
]

TRAINING_RESULTS_FILENAME: str = "metadata.json"
MODEL_FILENAME: str = "model.pt"

TEST_FILES: int = 256
VAL_FILES: int = 256

COLOR_MAP = {
    "autoencoder_val": "blue",
    "autoencoder_test": "midnightblue",
    "segan_val": "coral",
    "segan_test": "peru",
    "wavenet_val": "lime",
    "wavenet_test": "darkseagreen",
}
