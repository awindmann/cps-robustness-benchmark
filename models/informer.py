# Informer model.
# 
# Original paper: https://cdn.aaai.org/ojs/17325/17325-13-20819-1-2-20210518.pdf
# Based on the repository "Time-Series-Library" (https://github.com/thuml/Time-Series-Library) under the MIT License.

import torch
import torch.nn as nn
from models.utils.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from models.utils.SelfAttention_Family import ProbAttention, FullAttention, AttentionLayer
from models.utils.Embed import DataEmbedding
from models.base_module import BaseLitModule


class Informer(BaseLitModule):
    """Informer model.
    Args:
        d_model: int, dimension of the model
        d_ff: int, dimension of the feed-forward network
        n_layers_enc: int, number of encoder layers
        n_layers_dec: int, number of decoder layers
        n_heads: int, number of attention heads
        embed: str, type of embedding
        freq: str, frequency of the time series
        dropout: float, dropout rate
        factor: int, factor for ProbAttention
        activation: str, activation function
        loss: str, loss function
        distil: bool, whether to use distillation
    """
    def __init__(
        self, d_model=512, d_ff=512, n_layers_enc=3, n_layers_dec=2, n_heads=8, 
        embed="fixed", freq="s", dropout=0.0, factor=5, activation="gelu", distil=True,
        loss="MSE", **kwargs
    ):
        super().__init__(**kwargs)
        self.model_architecture = "Informer"
        
        # Embedding
        self.enc_embedding = DataEmbedding(self.d_features, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(self.d_features, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(n_layers_enc)
            ],
            [
                ConvLayer(d_model)
                for _ in range(n_layers_enc - 1)
            ] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(n_layers_dec)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.d_features, bias=True)
        )
        self.loss_fn = self.metrics[loss].clone()

    def _shared_step(self, x, y):
		# x: (batch_size, d_seq_in, d_features)
		# y: (batch_size, d_seq_out, d_features)

        enc_out = self.enc_embedding(x, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        x_dec = torch.zeros((x.size(0), self.d_seq_out, self.d_features), device=self.device)
        dec_out = self.dec_embedding(x_dec, None)

        y_pred = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        loss = self.loss_fn(y_pred, y) if y is not None else None
        return {
            "pred": y_pred,
            "target": y,
            "loss": loss,
        }
