import itertools
from datetime import datetime
import argparse
import os
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
# from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler  TODO profiler
import mlflow

import config as config
from data.data_module import TSDataModule
import models


torch.backends.cudnn.benchmark = False

def train_model(model_architecture, hparams, dataset, args):
    # model_hparam1_hparam2_...
    model_name = '_'.join([model_architecture] + [''.join([k[0] for k in key.split('_')]) + str(v) for key, v in hparams.items()])
    model_name = model_name.replace(".", "-").replace("[", "(").replace("]", ")").replace(", ", "-")  # remove special characters for MLFlow

    if args.logdir.startswith(("f", "i", "l", "e", ":")):  # TODO update when fixed
        raise ValueError("Lightning 2.4 does not accept logdirs starting with charactes in 'file:', see https://github.com/Lightning-AI/pytorch-lightning/issues/20279")

    # set seed
    pl.seed_everything(42, workers=True)

    # load datamodule
    dm = TSDataModule(
        file_path=dataset,
        input_len=args.input_len,
        target_len=args.target_len,
        n_train_samples=args.n_train_samples,
        n_val_samples=args.n_val_samples,
        train_split=args.train_split,
        val_split=args.val_split,
        purged_fraction=args.purged_fraction,
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        seed=args.seed,
        s3_endpoint=args.minio_endpoint
    )

    # configure pl trainer
    callbacks = list()
    if args.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                monitor=f"ep_val_loss", 
                filename='{epoch}-{ep_val_loss:.5f}',
                save_top_k=1,
                mode="min"
                )
            )
    if args.early_stopping:
        callbacks.append(EarlyStopping(monitor=f"ep_val_loss", patience=args.early_stopping_patience))
    dataset_name = dataset.split('/')[-1].split('.')[0]  # path/to/file.csv -> "file"
    tags = dict(
            model_architecture=model_architecture,
            dataset=dataset_name,
            date=datetime.today().strftime('%Y-%m-%d'),
            best_model=False
        )
    # without these changes mlflow was not able to send the hparams
    # to the remote tracking server. It failed creating proto messages
    tags.update(hparams)
    tags = {k:str(v) for k,v in tags.items()}

    os.environ["AWS_ENDPOINT_URL"] = args.minio_endpoint 
    logger = MLFlowLogger(
        tracking_uri=args.logdir if args.logdir.startswith("http://") else None,
        save_dir=args.logdir if not args.logdir.startswith("http://") else None,
        experiment_name=f"{args.mlflow_experiment_prefix}-{dataset_name}",
        run_name=model_name,
        tags=tags,
        log_model=True if args.save_checkpoint else False
    )
    if args.lr_scheduler:
        callbacks.append(LearningRateMonitor())
    # profiler = SimpleProfiler(filename="last-run-profiler-results")
    # profiler = AdvancedProfiler(filename="profiler-results")

    trainer = pl.Trainer(
        enable_checkpointing=True if args.save_checkpoint else False,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        # profiler=profiler
        )
    if trainer.strategy.root_device.type == "cuda":
        torch.set_float32_matmul_precision('medium')

    # load model
    try:
        model_class = getattr(models, model_architecture)
        model = model_class(
            d_features=dm.n_features, 
            d_seq_in=args.input_len,
            d_seq_out=args.target_len,
            lr_scheduler=args.lr_scheduler,
            **hparams
        )
    except AttributeError:
        raise ValueError(f"The model '{model_architecture}' does not exist.")

    trainer.fit(model=model, datamodule=dm)

    if args.save_checkpoint:
        # unfortunately, ModelCheckpoint does not interact well with MLFlowLogger, so we have to delete its checkpoint manually
        # MLFlowLogger saves its own checkpoint in the artifact directory, using the ModelCheckpoint specifications (monitor, mode, etc.)
        ckpt_path = callbacks[0]._ModelCheckpoint__resolve_ckpt_dir(trainer)
        shutil.rmtree(ckpt_path)
        for folder in [os.path.dirname(ckpt_path), os.path.dirname(os.path.dirname(ckpt_path))]:
            if os.path.exists(folder) and not os.listdir(folder):
                os.rmdir(folder)
    print("Done.")


def hyperparam_tuning(model_architecture, hparams, dataset, args=None):
    # process hparams
    # create a list of hparam keys that have direct values 
    direct_keys = [k for k, v in hparams.items() if not isinstance(v, list)]
    # create a list of hparam keys that have list values
    list_keys = [k for k, v in hparams.items() if isinstance(v, list)]
    # get the list values for each key that has a list value
    list_values = [hparams[k] for k in list_keys]
    
    num_combinations = 1
    for v in list_values:
        num_combinations *= len(v)
    print(f"Training the model with {num_combinations} different hyperparameter combinations.")

    # iterate over every combination of list values
    curr_combination = 0
    failed_runs = list()
    failed_runs_reason = list()
    for combo in itertools.product(*list_values):
        curr_combination += 1
        print("-"*80 + f"\nTraining '{model_architecture}' hyperparameter combination {curr_combination} of {num_combinations}.")
        # create a new dictionary with the combination of values
        combo_dict = dict(zip(list_keys, combo))
        # merge the new dictionary with the direct values
        merged_dict = {**{k: v for k, v in hparams.items() if k in direct_keys}, **combo_dict}
        print("Hyperparameters: ", merged_dict)
        if args is not None and not args.retrain:
            # check if already trained
            tracking_uri = args.logdir if args.logdir.startswith("http://") else "file:" + args.logdir
            mlflow.set_tracking_uri(tracking_uri)
            client = mlflow.MlflowClient()
            dataset_name = dataset.split('/')[-1].split('.')[0]  # path/to/file.csv -> "file"
            experiment = client.get_experiment_by_name(f"{args.mlflow_experiment_prefix}-{dataset_name}")
            if experiment is not None:
                model_name = '_'.join([model_architecture] + [''.join([k[0] for k in key.split('_')]) + str(v) for key, v in merged_dict.items()])
                model_name = model_name.replace(".", "-").replace("[", "(").replace("]", ")").replace(", ", "-")  # remove special characters for MLFlow
                filter_string = f"tags.mlflow.runName = '{model_name}' AND attributes.status = 'FINISHED' AND metrics.epoch > {args.early_stopping_patience}"
                old_runs = client.search_runs([experiment.experiment_id], filter_string, order_by=[f"metrics.best_val_loss {'ASC'}"])
                if len(old_runs) > 0:
                    start_time = datetime.fromtimestamp(old_runs[0].info.start_time / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Model '{model_name}' was already trained on {start_time} (Run ID: {old_runs[0].info.run_id}) using these same hyperparameters. Skipping this run.")
                    continue
        # try to run the training function with the merged dictionary
        try:
            train_model(model_architecture, hparams=merged_dict, dataset=dataset, args=args)
        except Exception as e:
            model_name = '_'.join([model_architecture] + [''.join([k[0] for k in key.split('_')]) + str(v) for key, v in merged_dict.items()])
            if args.raise_error:
                raise e
            elif "CUDA out of memory" in str(e):
                failed_runs.append(model_name)
                failed_runs_reason.append("CUDA out of memory")
                print("-"*80 + f"\nCUDA out of memory, skipping {model_name}.\nSkipped models: {str(failed_runs)}\n" + "-"*80)
            else:
                failed_runs.append(model_name)
                failed_runs_reason.append(str(e))
                print(str(e))
                print("-"*80 + f"\nUnknown error, skipping {model_name}.\nSkipped models: {str(failed_runs)}\n" + "-"*80)
    return failed_runs, failed_runs_reason


def train_all_models(args):
    """ Train all models. 
    Performs hyperparameter tuning for each model.
    """
    assert args.model.lower() in ["all"] + [model.lower() for model in config.HPARAMS.keys()], f"Model '{args.model}' not in {['all'] + list(config.HPARAMS.keys())}."
    failed_runs = list()
    failed_runs_reasons = list()
    if type(args.data_path) != list:
        args.data_path = list(args.data_path)
    for dataset in args.data_path:
        for model_architecture, hparams in config.HPARAMS.items():
            if args.model == "all" or args.model.lower() == model_architecture.lower():
                print(f"\nTraining '{model_architecture}' on {dataset} (Experiment: {args.mlflow_experiment_prefix}).")
                failed_runs_, failed_runs_reasons_ = hyperparam_tuning(model_architecture, hparams, dataset, args)
                failed_runs.extend(failed_runs_)
                failed_runs_reasons.extend(failed_runs_reasons_)
    if len(failed_runs) > 0:
        print(f"\nIn total, {len(failed_runs)} runs failed.")
        for run, reason in zip(failed_runs, failed_runs_reasons):
            print(f"\nRun {run} failed: {reason}")
    else:
        print("\nAll runs successful.")


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

    train_all_models(args)

