import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pretty_midi
from torch.utils.data import Dataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# Model
class Model(nn.Module):
    # Constructor:
    # Input:
    # - num_embeddings: This are the total number of pitches that we are gonna be embedding.
    # - embedding_dim: This are the dimension of the embedding, means how many dimension the embedding should be in.
    # - input_size: This is the total number of inputs the model could sustain.
    # - hidden_size: This are the number of hidden nodes in the neural network.
    # - num_layers: This are the number of layers of the hidden nodes, means how many layers does the neural network have.
    # - dropout: This is for the neural network not to memorize the training data.
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        input_size,
        hidden_size,
        num_layers,
        dropout,
    ):
        # Call the init function of the parent class
        super().__init__()

        # This is the Embedding model that would give us the embedding of the pitch.
        # Input:
        # - num_embeddings: Total number of values to embed, this are like total number of notes.
        # - embedding_dim: This are the dimension of the embedding graph.
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

        # This would give us the LSTM model that would be taking in the input and giving us the output,
        # Input:
        # - input_size: This is the size of input the model would be getting or the number of the notes the model would be getting
        # - hidden_size: This is the size of hidden nodes in the model that would be assigned the weights.
        # - num_layers: This is the number of layers the neural network would be having.
        # - batch_first: This is a boolean that tells that does the neural network gets to work with batch or not
        # - dropout: This is for the neural network memory so that he just doesn't remember but uses the logic too.
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.output_layer = nn.Linear(
            in_features=hidden_size, out_features=embedding_dim
        )

    def forward(self, X, hidden=None):
        # embedded is the embedding of the dataset that we are gonna take in.
        # Input:
        # - X: This is the dataset that it needs to embed
        embedded = self.embedding(X)

        # This output and the hidden stores the output given by the model and the hidden state of the model
        # Input:
        # - embedded: This the embedded version of the dataset
        # - hidden: This would be almost all the time None, because we don't assign any here.
        output, hidden = self.model(embedded, hidden)

        # This would make the output of the model according to the our output layer
        # Input:
        # - output: This is the output we got from the model.
        logit_output = self.output_layer(output)

        # This all the steps carry the data forward means the output logit output would be carring the data of the embedded, ouptut and the logit output all of them.
        return logit_output, hidden


if __name__ == "__main__":
    print("Hello World")
