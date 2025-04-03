import argparse
import os
import s3fs
import mlflow

import config as config



def clean(dataset, args):
    """Deletes model checkpoints from the logs. 
    Keeps the args.n_to_keep best checkpoints per model architecture. 
    The logged values are not touched by this function.
    """
    tracking_uri = args.logdir if args.logdir.startswith("http://") else "file:" + args.logdir
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    dataset_name = dataset.split('/')[-1].split('.')[0]  # path/to/file.csv -> "file"
    experiment = client.get_experiment_by_name(f"{args.mlflow_experiment_prefix}-{dataset_name}")
    if experiment is None:
        raise ValueError(f"No experiments with prefix {args.mlflow_experiment_prefix} on {dataset_name} found.")
    all_runs = client.search_runs([experiment.experiment_id], "")
    model_architectures = {run.data.tags["model_architecture"] for run in all_runs if "model_architecture" in run.data.tags}

    for model_architecture in model_architectures:
        if args.model == "all" or args.model.lower() == model_architecture.lower():
            print(f"\nCleaning logs of model '{model_architecture}' on dataset '{dataset_name}'.")
            # get best runs
            runs = client.search_runs(
                [experiment.experiment_id],
                f"tags.model_architecture = '{model_architecture}'",
                order_by=[f"metrics.best_val_loss {'ASC'}"],
            )
            if len(runs) < args.n_to_keep:
                print(f"{model_architecture}: Keep the remaining {len(runs)} checkpoints.")
                continue
            for run in runs[args.n_to_keep:]:
                s3 = s3fs.S3FileSystem(key=os.environ.get('AWS_ACCESS_KEY_ID'), secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), client_kwargs={"endpoint_url": args.minio_endpoint})
                checkpoint_path = s3.glob(run.info.artifact_uri + "/*/*.ckpt")
                # checkpoint_path = s3.glob(run.info.artifact_uri + "/model/checkpoints/*/*.ckpt")
                checkpoint_path = ["s3://" + path for path in checkpoint_path]
                if len(checkpoint_path) == 0:
                    continue
                else: 
                    checkpoint_path = checkpoint_path[0]
                    print(f"Deleting {checkpoint_path}")
                    s3.rm(checkpoint_path)


if __name__ == "__main__":
    # read args from config
    parser = argparse.ArgumentParser()
    for arg in dir(config):
        if arg.startswith("__"):
            continue
        arg_name = f"--{arg.lower().replace('_', '-')}"
        default_value = getattr(config, arg)
        if type(default_value) in [int, float, str]:
            parser.add_argument(arg_name, type=type(default_value), default=default_value, 
                                help=f"Set {arg} (default: {default_value})")
        elif type(default_value) is list:
            parser.add_argument(arg_name, nargs="*", default=default_value,
                                help=f"Set {arg} as a space-separated list (default: {default_value})")
    args = parser.parse_args()
    if type(args.data_path) != list:
        args.data_path = list(args.data_path)
    os.environ["AWS_ENDPOINT_URL"] = args.minio_endpoint

    # clean the logs for each dataset
    for i, dataset in enumerate(args.data_path):
        print(f"\nCleaning logs for dataset {i+1}/{len(args.data_path)}: {dataset}.")
        try:
            clean(dataset=dataset, args=args)
        except Exception as e:
            print(e)

