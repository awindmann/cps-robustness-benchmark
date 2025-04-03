import pandas as pd
import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    SymmetricMeanAbsolutePercentageError,
    MeanSquaredError,
)

from visualizations.plots import plot_performance_scores, plot_scenario_performance


class BaseLitModule(pl.LightningModule):
    """pytorch lightning core module
    This module is the base class for all models.
    It implements the training, validation and test steps.
    Args:
        d_features (int): number of features, can be found in data_module.nb_of_features
        d_seq_in (int): input sequence length
        d_seq_out (int): output sequence length
        test_dataloader_mapping (list): list mapping dataloader indices to scenarios. Only used for testing.
        lr (float): learning rate
        lr_scheduler (bool): use of learning rate scheduler
        beta1 (float): beta1 for Adam optimizer
        beta2 (float): beta2 for Adam optimizer
        eps (float): epsilon for Adam optimizer
    """
    def __init__(self, 
                 d_features, d_seq_in=250, d_seq_out=50,
                 test_dataloader_mapping=None,
                 lr=1e-3, lr_scheduler=False,
                 beta1=0.9, beta2=0.999, eps=1e-8,
                 **kwargs  #TODO remove, for backwards compatibility
                 ):
        super().__init__()

        self.save_hyperparameters()  # stores hyperparameters in self.hparams and allows logging and checkpointing
        self.d_features = d_features
        self.d_seq_in = d_seq_in
        self.d_seq_out = d_seq_out
        
        self.model_architecture = None
        self.example_input_array = torch.rand(1, self.d_seq_in, self.d_features)  # 1 as example batch size

        self.metrics = {
            "MAE": MeanAbsoluteError(),
            "MAPE": MeanAbsolutePercentageError(),
            "SMAPE": SymmetricMeanAbsolutePercentageError(),
            "MSE": MeanSquaredError(),
        }
        self.test_mode = False  # concerns logging in validation step. Set via self.set_test_mode()
        self.test_metric = None
        self.test_metric_fn = None
        
        self.test_dataloader_mapping = test_dataloader_mapping
        
        self.validation_step_outputs = list()
        self.test_step_outputs = list()
        self.min_epoch_val_loss = float("inf")

    def _shared_step(self, x, y):
        """Shared step used in training, validation and test step.
        Should return a dictionary containing at least a prediction "pred" and a loss "loss".
        """
        raise NotImplementedError("This should be implemented in the model that inherits from BaseLitModule.")

    @torch.no_grad()
    def forward(self, x):
        return self._shared_step(x, None)["pred"]
    
    def training_step(self, batch, batch_id):
        x, y = batch
        outputs = self._shared_step(x, y)
        self.log(f"train_loss", outputs["loss"], logger=True)
        return outputs["loss"]

    def validation_step(self, batch, batch_id):
        x, y = batch
        outputs = self._shared_step(x, y)
        if not self.test_mode:  # log custom loss
            loss = outputs["loss"]
            self.log(f"val_loss", loss, logger=True)
        else:  # log test metric
            loss = self.test_metric_fn(outputs["pred"], y)  # can differ from loss fn
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_losses = torch.stack(self.validation_step_outputs)
        mean_loss = torch.mean(epoch_losses)
        if not self.test_mode:  # log custom loss
            self.log(f"ep_val_loss", mean_loss, prog_bar=True, logger=True)
            # save if this is the best model so far
            if mean_loss < self.min_epoch_val_loss:
                self.min_epoch_val_loss = mean_loss
                self.log(f"best_val_loss", mean_loss, logger=True)
        else:  # log test metric
            self.log(f"{self.test_metric}_val", mean_loss, logger=True)
        self.validation_step_outputs.clear()  # free memory

    def set_test_mode(self, test_metric):
        """Set the test mode to True.
        Args:
            test_metric (str): name of the metric used for test step
        """
        assert test_metric in self.metrics, f"test_metric_name {test_metric} not in metrics. Choose from {list(self.metrics.keys())} or add implementation to BaseLitModule."
        self.test_metric = test_metric
        self.test_mode = True
        self.test_metric_fn = self.metrics[test_metric].clone()
        print(f"Test metric set to {test_metric}.")

    def set_test_dataloader_mapping(self, test_dataloader_mapping):
        """Set the test dataloader mapping.
        Args:
            test_dataloader_mapping (list): list mapping dataloader indices to scenarios
        """
        self.test_dataloader_mapping = test_dataloader_mapping
    
    def on_test_start(self):
        assert self.test_dataloader_mapping is not None, "test_dataloader_mapping must be set for testing"

    def test_step(self, batch, batch_id, dataloader_idx=0):
        x, y = batch
        outputs = self._shared_step(x, y)
        loss = self.test_metric_fn(outputs["pred"], y)
        self.test_step_outputs.append({
            "dataloader_idx": dataloader_idx,
            "loss": loss.item() if hasattr(loss, "item") else loss
        })

    def on_test_end(self):
        """Calculate performance metrics and log them."""
        result_df = pd.DataFrame(self.test_step_outputs)
        result_df = result_df.groupby(["dataloader_idx"]).agg(["mean"])  # aggregate over batches
        result_df.columns = ["loss"]
        result_df = pd.concat([self.test_dataloader_mapping, result_df], axis=1)

        # calculate performance
        normal_performance = result_df[result_df.scenario=="normal"]["loss"].values[0]  # performance on test dataset without disturbances
        self.logger.experiment.log_metric(self.logger.run_id, f"{self.test_metric}_test", normal_performance)
        result_df["rel_perf"] = (1e-6 + normal_performance) / (1e-6 + result_df["loss"])
        # self.logger.experiment.log_dict(self.logger.run_id, result_df.reset_index().to_dict(orient="list"), artifact_file=f"all_test_results_{self.test_metric}.json")
        fig = plot_scenario_performance(result_df, self.test_metric, self.model_architecture)
        self.logger.experiment.log_figure(self.logger.run_id, fig, artifact_file=f"scenario_rel_perf_{self.test_metric}.pdf")

        # calculate robustness score
        rel_perf_df = result_df[result_df.scenario != "normal"][["scenario", "rel_perf"]].groupby("scenario").agg(["mean", "std"])
        rel_perf_df.columns = ["mean", "std"]
        self.logger.experiment.log_dict(self.logger.run_id, rel_perf_df.reset_index().to_dict(orient="list"), artifact_file=f"scenario_rel_perf_{self.test_metric}.json")
        self.logger.experiment.log_metric(self.logger.run_id, f"robustness_score_min_{self.test_metric}", rel_perf_df["mean"].min())
        self.logger.experiment.log_metric(self.logger.run_id, f"robustness_score_mean_{self.test_metric}", rel_perf_df["mean"].mean())
        self.logger.experiment.log_metric(self.logger.run_id, f"robustness_score_prod_{self.test_metric}", rel_perf_df["mean"].prod())
        
        # free memory
        self.test_step_outputs = pd.DataFrame(columns=["dataloader_idx", "loss"])
        del result_df, normal_performance, rel_perf_df, #severity_stats, performance_scores TODO remove if deleted

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), 
            lr=self.hparams.lr,  
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.eps
            )
        if self.hparams.lr_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=25, min_lr=1e-5)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "ep_val_loss"}]
        return optimizer
