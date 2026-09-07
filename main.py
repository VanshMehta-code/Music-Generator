import numpy as np
import pandas as pd
import pretty_midi
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset


# Model
class Model(nn.Module):
    def __init__(
        self,
        num_pitches,
        num_velocity,
        num_duration,
        num_conditions,
        pitch_embedding_dim,
        velocity_embedding_dim,
        duration_embedding_dim,
        hidden_size,
        num_layers,
        dropout,
        seq_len,
    ):
        super().__init__()
        self.seq_len = seq_len
        input_size = (
            pitch_embedding_dim + velocity_embedding_dim + duration_embedding_dim
        )
        self.pitch_embedding = nn.Embedding(
            num_embeddings=num_pitches, embedding_dim=pitch_embedding_dim
        )
        self.duration_embedding = nn.Embedding(
            num_embeddings=num_duration, embedding_dim=duration_embedding_dim
        )
        self.velocity_embedding = nn.Embedding(
            num_embeddings=num_velocity, embedding_dim=velocity_embedding_dim
        )
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.pitch_output_layer = nn.Linear(
            in_features=hidden_size, out_features=pitch_embedding_dim
        )
        self.velocity_output_layer = nn.Linear(
            in_features=hidden_size, out_features=velocity_embedding_dim
        )
        self.duration_output_layer = nn.Linear(
            in_features=hidden_size, out_features=duration_embedding_dim
        )

    def forward(self, X_pitch, X_duration, X_velocity, X_cond, hidden=None):
        batch, seq_len = X_pitch.shape
        embedded = torch.cat(
            [
                self.pitch_embedding(X_pitch),
                self.duration_embedding(X_duration),
                self.velocity_embedding(X_velocity),
            ],
            dim=-1,
        )
        cond = X_cond.unsqueeze(1).repeat(1, seq_len, 1)
        input = torch.cat([embedded + cond], dim=-1)

        output, hidden = self.model(input, hidden)

        logit_pitch_output = self.pitch_output_layer(output)
        logit_duration_output = self.duration_output_layer(output)
        logit_velocity_output = self.velocity_output_layer(output)

        return (
            logit_pitch_output,
            logit_duration_output,
            logit_velocity_output,
        ), hidden


if __name__ == "__main__":
    print("Hello World")
