import argparse
import glob
import os
import tempfile
import s3fs
import pandas as pd
from scipy.stats import pearsonr
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler  #TODO profiler

import models
from data.data_module import TSDataModule
from visualizations.plots import plot_scenario_performance_hist
import config as config


torch.backends.cudnn.benchmark = False

def test_on_dataset(dataset, args):
    # set seed
    pl.seed_everything(42, workers=True)
    # load datamodule
    dm = TSDataModule(
        file_path=dataset,
        input_len=args.input_len,
        target_len=args.target_len,
        n_train_samples=args.n_train_samples,
        n_val_samples=args.n_val_samples,
        n_test_samples=args.n_test_samples,
        n_severity_levels=args.n_severity_levels,
        prct_affected_sensors=args.prct_affected_sensors,
        train_split=args.train_split,
        val_split=args.val_split,
        purged_fraction=args.purged_fraction,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        seed=args.seed
    )
    # load best model
    tracking_uri = args.logdir if args.logdir.startswith("http://") else "file:" + args.logdir
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    dataset_name = dataset.split('/')[-1].split('.')[0]  # path/to/file.csv -> "file"
    experiment = client.get_experiment_by_name(f"{args.mlflow_experiment_prefix}-{dataset_name}")
    if experiment is None:
        raise ValueError(f"No experiments with prefix {args.mlflow_experiment_prefix} on {dataset_name} found.")
    all_runs = client.search_runs([experiment.experiment_id], max_results=10000)  
    n_runs = len(all_runs)
    model_architectures = {run.data.tags["model_architecture"] for run in all_runs if "model_architecture" in run.data.tags}
    print(f"Found a total of {n_runs} runs with {len(model_architectures)} different model architectures.")

    for model_architecture in model_architectures:
        if args.model == "all" or args.model.lower() == model_architecture.lower():
            print(f"\nTesting model '{model_architecture}' on dataset '{dataset_name}'.")
            # get best run
            runs = client.search_runs(
                [experiment.experiment_id],
                f"tags.model_architecture = '{model_architecture}'",
                order_by=[f"metrics.best_val_loss {'ASC'}"],
            )
            if len(runs) == 0:
                print(f"No runs found for model '{model_architecture}' in experiment {dataset_name} (id: {experiment.experiment_id}).")
                continue
            best_run = runs[0]
            if (  # we don't want to retest and the best model has already been tested
                not args.retest 
                and best_run.data.tags.get("best_model") == "true"
                and f"{args.test_metric}_test" in best_run.data.metrics
            ):  
                print(f"{model_architecture} has already been tested. Skipping.")
                continue
            client.set_tag(best_run.info.run_id, "best_model", "true")
            for run in runs[1:]:
                client.set_tag(run.info.run_id, "best_model", "false")

            # get checkpoint path
            ckpt_root = "/model/checkpoints"
            if best_run.info.artifact_uri.startswith("s3://"):
                s3 = s3fs.S3FileSystem(key=os.environ.get('AWS_ACCESS_KEY_ID'), secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), client_kwargs={"endpoint_url": args.minio_endpoint})
                checkpoint_path = s3.glob(best_run.info.artifact_uri + f"{ckpt_root}/*/*.ckpt")
                checkpoint_path = ["s3://" + path for path in checkpoint_path]
                defaul_root_dir = "/".join(best_run.info.artifact_uri.split("/")[:-2]) + "/"
            else:
                checkpoint_path = glob.glob(f"{best_run.info.artifact_uri}{ckpt_root}/*/*.ckpt".replace("file://", ""))
                defaul_root_dir = os.getcwd()

            if len(checkpoint_path) == 0:
                ckpt_root = ""  # ckpt can also be saved directly in artifacts
                if best_run.info.artifact_uri.startswith("s3://"):
                    s3 = s3fs.S3FileSystem(key=os.environ.get('AWS_ACCESS_KEY_ID'), secret=os.environ.get('AWS_SECRET_ACCESS_KEY'), client_kwargs={"endpoint_url": args.minio_endpoint})
                    checkpoint_path = s3.glob(best_run.info.artifact_uri + f"{ckpt_root}/*/*.ckpt")
                    checkpoint_path = ["s3://" + path for path in checkpoint_path]
                    defaul_root_dir = "/".join(best_run.info.artifact_uri.split("/")[:-2]) + "/"
                else:
                    checkpoint_path = glob.glob(f"{best_run.info.artifact_uri}{ckpt_root}/*/*.ckpt".replace("file://", ""))
                    defaul_root_dir = os.getcwd()

            if len(checkpoint_path) == 0:  # still nothing
                raise ValueError(f"No checkpoint found for model '{model_architecture}' in experiment {dataset_name} (id: {experiment.experiment_id}).")
            checkpoint_path = checkpoint_path[0]

            # load model
            model_class = getattr(models, model_architecture)
            if best_run.info.artifact_uri.startswith("s3://"):
                with tempfile.NamedTemporaryFile() as temp_file:  # workaround, due to permissions
                    with s3.open(checkpoint_path, 'rb') as s3_file:
                        temp_file.write(s3_file.read())
                        temp_file.flush()
                    model = model_class.load_from_checkpoint(temp_file.name)
            else: 
                model = model_class.load_from_checkpoint(checkpoint_path)
            model.set_test_mode(args.test_metric)
            model.set_test_dataloader_mapping(dm.test_dataloader_mapping)
            assert args.input_len == model.hparams.d_seq_in, f"Model input length ({model.hparams.d_seq_in}) does not match input length in config ({args.input_len})."
            assert args.target_len == model.hparams.d_seq_out, f"Model prediction length ({model.hparams.d_seq_out}) does not match target length in config ({args.target_len})."
            
            # configure pl trainer
            logger = MLFlowLogger(
                    tracking_uri=args.logdir if args.logdir.startswith("http://") else "file:" + args.logdir,
                    save_dir=args.logdir,
                    experiment_name=f"{dataset_name}",
                    run_id=best_run.info.run_id
                )
            trainer = pl.Trainer(
                enable_checkpointing=True if args.save_checkpoint else False,
                max_epochs=args.max_epochs,
                accelerator=args.accelerator,
                devices=args.devices,
                precision=args.precision,
                log_every_n_steps=args.log_every_n_steps,
                logger=logger,
                default_root_dir=defaul_root_dir,
                # profiler=AdvancedProfiler(filename="0/profiler-results")
                )

            if trainer.strategy.root_device.type == "cuda":
                torch.set_float32_matmul_precision('medium')

            trainer.validate(model=model, datamodule=dm, verbose=False)
            trainer.test(model=model, datamodule=dm)
    print("Done.")
    return n_runs


def meta_analysis(args):
    print("Running meta analysis.")
    tracking_uri = args.logdir if args.logdir.startswith("http://") else "file:" + args.logdir
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    experiments = client.search_experiments()
    experiments = [exp for exp in experiments if exp.name.startswith(args.mlflow_experiment_prefix)]
    datasets = [dataset.split('/')[-1].split('.')[0] for dataset in args.data_path]

    data = list()
    scenarios = list()
    for experiment in experiments:
        dataset_name = experiment.name.split(args.mlflow_experiment_prefix)[-1][1:]
        if dataset_name not in datasets:
            continue
        best_runs = client.search_runs([experiment.experiment_id], "tags.best_model = 'true'")
        for run in best_runs:
            model_architecture = run.data.tags.get("model_architecture", None)
            val_metric = run.data.metrics.get(f"{args.test_metric}_val", None)
            test_metric = run.data.metrics.get(f"{args.test_metric}_test", None)
            robustness_score_min = run.data.metrics.get(f"robustness_score_min_{args.test_metric}", None)
            robustness_score_mean = run.data.metrics.get(f"robustness_score_mean_{args.test_metric}", None)
            robustness_score_prod = run.data.metrics.get(f"robustness_score_prod_{args.test_metric}", None)
            rel_perf_dict = pd.read_json(client.download_artifacts(run.info.run_id, f"scenario_rel_perf_{args.test_metric}.json")).to_dict(orient="list")
            rel_perf_dict = {key: value for (key, value) in zip(rel_perf_dict["scenario"], rel_perf_dict["mean"])}
            scenarios = list(rel_perf_dict.keys()) if len(rel_perf_dict.keys()) > len(scenarios) else scenarios  # there are some scenarios that only apply for datasets with discrete sensors/actuators
            data.append({
                "dataset": dataset_name,
                "model": model_architecture,
                f"{args.test_metric}_val": val_metric,
                f"{args.test_metric}_test": test_metric,
                f"robustness_score_min_{args.test_metric}": robustness_score_min,
                f"robustness_score_mean_{args.test_metric}": robustness_score_mean,
                f"robustness_score_prod_{args.test_metric}": robustness_score_prod,
                **rel_perf_dict
            })
    result_df = pd.DataFrame(data, columns=[key for key in list(data[0].keys()) if key not in scenarios])  # no need for the relative performance per scenario here
    nan_rows = result_df[result_df.isnull().any(axis=1)]
    if not nan_rows.empty:
        for idx, row in nan_rows.iterrows():
            print(f"{row['model']} failed on {row['dataset']}.")
        print("Removing failed runs from computation.")
        result_df.dropna(inplace=True)
    model_results_df = result_df.drop(columns="dataset").groupby(["model"]).agg(["mean", "std"])
    model_results_df.columns = ['_'.join(col).strip() for col in model_results_df.columns]
    model_results_df.insert(0, "count", result_df.groupby("model").size())
    data_results_df = result_df.drop(columns="model").groupby(["dataset"]).agg(["mean", "std"])
    data_results_df.columns = ['_'.join(col).strip() for col in data_results_df.columns]
    data_results_df.insert(0, "count", result_df.groupby("dataset").size())

    scenario_df = pd.DataFrame(data, columns=["dataset", "model"] + scenarios)
    hist_plot = plot_scenario_performance_hist(scenario_df, args.test_metric)

    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print(model_results_df)
    print(data_results_df)

    # Compute correlation between val->test drop and robustness score
    result_df["val_to_test_drop_abs"] = result_df[f"{args.test_metric}_val"] - result_df[f"{args.test_metric}_test"]
    result_df["val_to_test_drop"] = result_df["val_to_test_drop_abs"] / result_df[f"{args.test_metric}_val"]
    # compute per-model correlations and p-values if possible
    def compute_corr(data):  # helper function
        count = len(data)
        if count > 2:
            correlation, p_value = pearsonr(
                data["val_to_test_drop"],
                data[f"robustness_score_{version}_{args.test_metric}"]
            )
        else:
            correlation, p_value = None, None
        return pd.Series({
            "count": count,
            "correlation": correlation,
            "p_value": p_value
        })
    # model correlation
    model_corr_per_mode = []
    for version in ["min", "mean", "prod"]:
        corr = compute_corr(result_df)
        correlation_df = pd.DataFrame([{
            "model": "all",
            "count": corr["count"],
            "correlation": corr["correlation"],
            "p_value": corr["p_value"]
        }]).set_index("model")
        model_corr_df = result_df.set_index("model").groupby("model").apply(compute_corr)
        correlation_df = pd.concat([correlation_df, model_corr_df]).astype({'count': 'int32'})
        model_corr_per_mode.append(correlation_df)
        print(f"\nModel correlation of {version}.")
        print(correlation_df)
    # dataset correlation
    data_corr_per_mode = []
    for version in ["min", "mean", "prod"]:
        corr = compute_corr(result_df)
        correlation_df = pd.DataFrame([{
            "dataset": "all",
            "count": corr["count"],
            "correlation": corr["correlation"],
            "p_value": corr["p_value"]
        }]).set_index("dataset")
        data_corr_df = result_df.set_index("dataset").groupby("dataset").apply(compute_corr)
        correlation_df = pd.concat([correlation_df, data_corr_df]).astype({'count': 'int32'})
        data_corr_per_mode.append(correlation_df)
        print(f"\nDataset correlation of {version}.")
        print(correlation_df)

    # log
    mlflow.set_experiment("Meta Analysis")
    with mlflow.start_run(run_name=f"{args.mlflow_experiment_prefix}-{args.test_metric}"):
        mlflow.log_dict(result_df.reset_index().to_dict(orient="list"), f"full_results_{args.test_metric}.json")
        mlflow.log_dict(model_results_df.reset_index().to_dict(orient="list"), f"model_results_{args.test_metric}.json")
        mlflow.log_dict(data_results_df.reset_index().to_dict(orient="list"), f"dataset_results_{args.test_metric}.json")
        mlflow.log_dict(scenario_df.reset_index().to_dict(orient="list"), f"scenario_results_{args.test_metric}.json")
        mlflow.log_figure(hist_plot, artifact_file=f"scenario_rel_perf_{args.test_metric}.pdf")
        for version, df in zip(["min", "mean", "prod"], model_corr_per_mode):
            mlflow.log_dict(df.reset_index().to_dict(orient="list"), f"model_correlations_{args.test_metric}_{version}.json")
        for version, df in zip(["min", "mean", "prod"], data_corr_per_mode):
            mlflow.log_dict(df.reset_index().to_dict(orient="list"), f"data_correlations_{args.test_metric}_{version}.json")
    
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for arg in dir(config):
        if arg.startswith("__"):
            continue
        arg_name = f"--{arg.lower().replace('_', '-')}"
        default_value = getattr(config, arg)
        if type(default_value) in [int, float, str]:
            parser.add_argument(arg_name, type=type(default_value), default=default_value, 
                                help=f"Set {arg} (default: {default_value})")
        elif type(default_value) is bool:
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, default=default_value,
                                help=f"Enable or disable {arg} (default: {default_value})")
        elif type(default_value) is list:
            parser.add_argument(arg_name, nargs="*", default=default_value,
                                help=f"Set {arg} as a space-separated list (default: {default_value})")
    args = parser.parse_args()

    os.environ["AWS_ENDPOINT_URL"] = args.minio_endpoint 

    if type(args.data_path) != list:
        args.data_path = list(args.data_path)
    n_runs = 0
    for i, dataset in enumerate(args.data_path):
        print(f"\nTesting dataset {i+1}/{len(args.data_path)}: {dataset}.")
        try:
            n_runs += test_on_dataset(dataset=dataset, args=args)
        except Exception as e:
            print(f"Error during testing of dataset {dataset}: {str(e)}")
    print(f"There are a total of {n_runs} models that have been trained.\n")

    if args.meta_analysis:
        meta_analysis(args)
