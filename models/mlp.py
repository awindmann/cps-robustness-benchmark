import torch.nn as nn

from models.base_module import BaseLitModule


class MLP(BaseLitModule):
    """Multi-layer perceptron model.
    Args:
        d_hidden_layers (list): list of hidden dimensions
        batch_norm (bool): whether to use batch normalization
        loss (str): the loss function to be used
    """
    def __init__(self, d_hidden_layers=None, batch_norm=False, loss="MSE", **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "MLP"

        if d_hidden_layers is None:
            d_hidden_layers = [128, 64]

        self.loss_fn = self.metrics[loss].clone()
        self.model = self._build_model(d_hidden_layers, batch_norm)

    def _build_model(self, d_hidden_layers, batch_norm):
        layers = []
        d_input = self.d_seq_in * self.d_features
        for d_hidden in d_hidden_layers:
            layers.append(nn.Linear(d_input, d_hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(d_hidden))
            layers.append(nn.ReLU())
            d_input = d_hidden
        # last layer
        layers.append(nn.Linear(d_hidden_layers[-1], self.d_seq_out * self.d_features))

        return nn.Sequential(*layers)

    def _shared_step(self, x, y):
        # x1 = x, x2 = y
        b_size = x.size(0)
        x = x.reshape(b_size, -1)

        y_pred = self.model(x)
        y_pred = y_pred.view(b_size, self.d_seq_out, self.d_features)

        loss = self.loss_fn(y_pred, y) if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
