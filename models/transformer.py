# Transformer model.
# 
# Original paper: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# Based on the repository "Time-Series-Library" (https://github.com/thuml/Time-Series-Library) under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from models.utils.SelfAttention_Family import FullAttention, AttentionLayer
from models.utils.Embed import DataEmbedding
import numpy as np

from models.base_module import BaseLitModule


class Transformer(BaseLitModule):

    def __init__(self, d_model=512, d_ff=2048, n_layers_enc=2, n_layers_dec=2, n_heads=8, embed="fixed", freq="s", dropout=0.1, factor=1, activation="gelu", loss="MSE", **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "Transformer"

        # Embedding
        self.enc_embedding = DataEmbedding(self.d_features, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(self.d_features, d_model, embed, freq, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(n_layers_enc)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(n_layers_dec)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, self.d_features, bias=True)
        )
        # Loss
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
