from kfp import dsl
from typing import List, Dict, Any, NamedTuple, Tuple
from kfp import kubernetes
from itertools import product

# define components
TRAINING_IMAGE = "gitlab.kiss.space.unibw-hamburg.de:4567/kiss/robust-ai-validation:v31"
KFP_PARALLELISM = 7


@dsl.component()
def compute_combinations(
    model_list: List, data_path_list: List
) -> List[Dict[str, str]]:
    from itertools import product

    combinations = [
        {"model": model, "data_path": path}
        for model, path in product(model_list, data_path_list)
    ]
    return combinations


@dsl.component()
def get_param_entries(
    param_dct: Dict[str, str],
) -> NamedTuple("outputs", data_path=str, model=str):
    outputs = NamedTuple("outputs", data_path=str, model=str)
    return outputs(param_dct["data_path"], param_dct["model"])


@dsl.container_component
def hparam_tuner_comp(
    data_path: str,
    model: str,
    max_epochs: int,
    mlflow_experiment_prefix: str,
):
    """Performs the hparam tuning for on dataset and one model"""
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["python", "run_training.py"],
        args=[
            "--data-path",
            data_path,
            "--model",
            model,
            "--max-epochs",
            max_epochs,
            "--mlflow-experiment-prefix",
            mlflow_experiment_prefix,
        ],
    )


# helper function
def add_minio_env_vars_to_tasks(task_list: List[dsl.PipelineTask]) -> None:
    """Adds environment variables for minio to the tasks"""
    for task in task_list:
        kubernetes.use_secret_as_env(
            task,
            secret_name="s3creds",
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            },
        )


@dsl.pipeline
def training_pipelne(
    model_architecture_list: List[str],
    data_path_list: List[str],
    max_epochs: int,
    mlflow_experiment_prefix: str,
) -> None:
    get_combinations_task = compute_combinations(
        model_list=model_architecture_list,
        data_path_list=data_path_list,
    )

    with dsl.ParallelFor(
        items=get_combinations_task.output,
        parallelism=KFP_PARALLELISM,
    ) as comb_ls_entry:
        get_params_task = get_param_entries(param_dct=comb_ls_entry)
        hparam_task = hparam_tuner_comp(
            model=get_params_task.outputs["model"],
            data_path=get_params_task.outputs["data_path"],
            max_epochs=max_epochs,
            mlflow_experiment_prefix=mlflow_experiment_prefix,
        )
        hparam_task.set_accelerator_type("nvidia.com/gpu").set_gpu_limit("1")
        add_minio_env_vars_to_tasks([hparam_task])


from kfp.client import Client


pipeline_args = dict(
    model_architecture_list=[
        "Transformer",
        "MLP",
        "LSTM",
        "GRU",
        "TCN",
        "Mamba",
        "Informer",
        "DLinear",
        "RIM"
    ],
    data_path_list=[
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
    ],
    max_epochs=1000,
    mlflow_experiment_prefix="param-test-02",
)


client = Client()
client.create_run_from_pipeline_func(
    training_pipelne,
    arguments=pipeline_args,
    experiment_name="param-test-02",
)
