### TRAINING ARGUMENTS ###
# run_training.py

DATA_PATH = [
    "s3://alexander-windmann-data/ETTh1.csv",
    "s3://alexander-windmann-data/ETTm1.csv",
    "s3://alexander-windmann-data/electricity.csv",
    "s3://alexander-windmann-data/SKAB_anomaly_free.csv",
    "s3://alexander-windmann-data/water_quality.csv",
    "s3://alexander-windmann-data/SWaT_Dataset_Normal_v1.parquet",

    # "s3://alexander-windmann-data/damadics.csv",
    # "s3://alexander-windmann-data/WADI_14days.csv",
    # "s3://alexander-windmann-data/three_tank_data.csv",
    # "s3://alexander-windmann-data/ETTh2.csv",
    # "s3://alexander-windmann-data/ETTm2.csv",
]
LOGDIR = "http://mlflow-server.alexander-windmann.svc.cluster.local"
MINIO_ENDPOINT = "http://minio.minio"
# MLFLOW_EXPERIMENT_PREFIX = "test-09"
MLFLOW_EXPERIMENT_PREFIX = "param-test-02"
LOG_EVERY_N_STEPS = 1
SAVE_CHECKPOINT = True
RETRAIN = False  # whether to retrain a model if that hparam combination on that dataset has already be trained

MAX_EPOCHS = 1000
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER = False  # if True, ReduceLROnPlateau is used
N_TRAIN_SAMPLES = 10000
N_VAL_SAMPLES = 1000
INPUT_LEN = 90
TARGET_LEN = 30
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
PURGED_FRACTION = 0.01  # fraction of samples to be purged from the start of the validation and test set to avoid leakage

BATCH_SIZE = 64
PRECISION = 32
NUM_WORKERS = 0
ACCELERATOR = "auto"
DEVICES = 1  # for benchmarking for research papers, use 1 device only
SEED = 42

# Hyperparameter Tuning
MODEL = "all"  # "all", "Mamba", "DLinear", "Informer", "Transformer", "GRU", "LSTM", "MLP", "TCN", "RIMs",
RAISE_ERROR = False  # if False, continue with next model if an error occurs during hyperparameter tuning
# HPARAMS contains the hyperparameter combinations to be tested for each model (via grid search).
# If MODEL is set to a specific model, only the hyperparameters for that model are used.
HPARAMS = dict(
    Mamba = dict(
        d_model=[32, 64, 128, None],
        d_conv=[4, 8],
        n_mamba_blocks=[2, 4],
        d_state=[32, 64, 128],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    Transformer = dict(
        d_model=[256, 512],
        d_ff=[1024, 2048],
        n_layers_enc=[2, 4],
        n_layers_dec=[2, 4],
        n_heads=[4, 8],
        dropout=[0, 0.1],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    Informer = dict(
        d_model=[256, 512],
        d_ff=[512, 1024, 2048],
        n_layers_enc=[2, 4],
        n_layers_dec=[2, 4],
        n_heads=[4, 8],
        dropout=[0, 0.1],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    GRU = dict(
        d_hidden=[64, 128, 256, 512],
        n_layers=[1, 2],
        autoregressive=[False, True],
        dropout=[0, 0.1],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    LSTM = dict(
        d_hidden=[64, 128, 256, 512],
        n_layers=[1, 2],
        autoregressive=[False, True],
        dropout=[0, 0.1],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    MLP = dict(
        d_hidden_layers=[
            [128], [256], [512], [1024],
            [128, 128], [256, 256], [512, 512],
            [128, 256, 128], [256, 512, 256], [512, 1024, 512]
            ],
        batch_norm=[False, True],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    DLinear = dict(
        moving_avg=[9, 13, 17, 21, 25, 29, 33],
        individual=[False, True],
        init_weights=[False, True],
        lr=[1e-3, 1e-4, 5e-4],
    ),
    TCN = dict(
        kernel_size=[3, 5, 7, 9],
        num_channels= [[16, 32, 32], [32, 64, 64], [64, 128, 128], [128, 256, 256]],
        dropout=[0, 0.1, 0.2],
        lr=[1e-3, 1e-4, 5e-4],
        ),
    RIMs = dict(
        d_hidden=[128, 256, 512],
        n_units=[4, 6, 8],
        n_active_RIMs=[2, 4, 6, 8],
        rnn_cell=["LSTM", "GRU"],
        autoregressive=[False, True],
        lr=[1e-3, 1e-4, 5e-4],
    )
)


### TESTING ARGUMENTS ###
# run_testing.py

N_SEVERITY_LEVELS = 101  # nb of dataloaders per scenario
N_TEST_SAMPLES = 128  # per dataloader. Total nb of test samples = nb of scenarios * N_SEVERITY_LEVELS * N_TEST_SAMPLES
PRCT_AFFECTED_SENSORS = 0.1
RETEST = True  # whether to retest models that have already been tested
META_ANALYSIS = True
TEST_METRIC = "MSE"  # "MAE", "MAPE", "SMAPE", "MSE",


### LOG CLEANING ARGUMENTS ###
# run_cleaning.py

N_TO_KEEP = 25  # keeps the n best runs per model architecture per dataset