import torch
import torch.nn as nn

from models.base_module import BaseLitModule



class GRU(BaseLitModule):
    """GRU model.
    Args:
        d_hidden (int): hidden dimension
        n_layers (int): number of layers
        bidirectional (bool): whether to use bidirectional GRU
        dropout (float): dropout rate
        autoregressive (bool): whether to predict one output at a time
        loss (str): name of the loss function, defaults to MSE
    """
    def __init__(self, d_hidden, n_layers=1, bidirectional=False, dropout=0.5, autoregressive=False, loss="MSE", **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "GRU"

        self.gru = nn.GRU(
            input_size=self.d_features,
            hidden_size=d_hidden,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        if not autoregressive:
            self.fc = nn.Linear(d_hidden * (bidirectional + 1), self.d_features * self.d_seq_out)
        else:
            self.fc = nn.Linear(d_hidden * (bidirectional + 1), self.d_features)
        self.autoregressive = autoregressive

        self.loss_fn = self.metrics[loss].clone()

    def _shared_step(self, x, y):
        b_size = x.size(0)

        if not self.autoregressive:
            _, h = self.gru(x)
            h = h[-1, :, :]
            y_pred = self.fc(h).view(b_size, self.d_seq_out, self.d_features)
        else:
            y_pred = []
            _, h = self.gru(x)
            output = self.fc(h[-1, :, :]).unsqueeze(1)
            y_pred.append(output)
            for _ in range(self.d_seq_out - 1):
                _, h = self.gru(output, h)
                output = self.fc(h[-1, :, :]).unsqueeze(1)
                y_pred.append(output)
            y_pred = torch.cat(y_pred, dim=1)
        
        loss = self.loss_fn(y_pred, y) if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }