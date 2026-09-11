import glob

import pretty_midi as midi
import torch
from torch import nn
from torch.nn import LSTM
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam


class Datautils:
    def __init__(self, file_path, max_limit=None) -> None:
        self.dir_file_path = file_path
        self.max_load_limit = max_limit

    def extract_notes(self, file):
        pm = midi.PrettyMIDI(file)
        notes = [
            (n.pitch, n.velocity, n.get_duration())
            for inst in pm.instruments
            for n in inst.notes
        ]
        return notes

    def extract_features(self, note):
        return (note.pitch, note.velocity, (note.end - note.start))

    def load_corpus(self):
        sequence = []
        files = glob.glob(f"{self.dir_file_path}/**/*.mid", recursive=True)
        files.extend(glob.glob(f"{self.dir_file_path}/**/*.midi", recursive=True))
        if self.max_load_limit is not None:
            files = files[: self.max_load_limit]
        for file in files:
            notes = self.extract_notes(file)
            sequence.append(notes)
        return sequence


class Datacorpus(Dataset):
    def __init__(self, sequence, seq_len) -> None:
        self.seq_len = seq_len
        self.example = []
        for seq in sequence:
            for i in range(len(seq) - seq_len):
                x = seq[i : i + seq_len]
                y = seq[i + 1 : i + seq_len + 1]
                self.example.append((x, y))

    def __getitem__(self, idx):
        x, y = self.example[idx]
        pitch_input = torch.tensor([n[0] for n in x], dtype=torch.long)
        cont_input = torch.tensor([[n[1], n[2]] for n in x], dtype=torch.float)
        pitch_output = torch.tensor([n[0] for n in y], dtype=torch.long)
        cont_output = torch.tensor([[n[1], n[2]] for n in y], dtype=torch.float)
        return (pitch_input, cont_input, pitch_output, cont_output)

    def __len__(self):
        return len(self.example)


class Model(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_pitches,
        pitch_embedding_dim,
        cont_in_features,
        cont_out_features,
        dropout,
    ) -> None:
        super().__init__()
        self.pitch_embedding = nn.Embedding(num_pitches, pitch_embedding_dim)
        self.cont_proj = nn.Linear(cont_in_features, cont_out_features)
        input_size = pitch_embedding_dim + cont_out_features
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.pitch_output = nn.Linear(hidden_size, num_pitches)
        self.cont_output = nn.Linear(hidden_size, cont_in_features)

    def forward(self, pitch, cont, hidden=None):
        pitch_input = self.pitch_embedding(pitch)
        cont_input = self.cont_proj(cont)
        input = torch.cat([pitch_input, cont_input], -1)

        output, hidden = self.lstm(input, hidden)

        pitch_logit = self.pitch_output(output)
        cont_logit = self.cont_output(output)

        return pitch_logit, cont_logit, hidden


class Train:
    def __init__(
        self,
        file_path="./data/",
        max_files=None,
        seq_len=32,
        batch_size=64,
        hidden_size=512,
        num_layers=4,
        num_pitches=128,
        pitch_embedding_dim=32,
        cont_in_features=2,
        cont_out_features=16,
        dropout=0.2,
        epochs=8,
        save_path="model.pt",
    ):
        self.num_pitches = num_pitches
        self.save_path = save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"INFO: device={self.device}")
        self.datautils = Datautils(file_path, max_files)
        sequence = self.datautils.load_corpus()
        print("INFO: Sequence created")
        print(f"INFO: sequence length: {len(sequence)}")
        self.datacorpus = Datacorpus(sequence, seq_len)
        print("INFO: Dataset Loaded")
        self.dataloader = DataLoader(self.datacorpus, batch_size, True)
        print("INFO: Data Loader is ready")
        print(f"INFO: Data Loader Length: {len(self.dataloader)}")
        self.model = Model(
            hidden_size,
            num_layers,
            num_pitches,
            pitch_embedding_dim,
            cont_in_features,
            cont_out_features,
            dropout,
        ).to(self.device)
        print("INFO: Model Created.")
        self.optimize = Adam(self.model.parameters())
        self.epochs = epochs
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse_error = nn.MSELoss()

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for pitch_input, cont_input, pitch_output, cont_output in self.dataloader:
                pitch_input, cont_input = (
                    pitch_input.to(self.device),
                    cont_input.to(self.device),
                )
                pitch_output, cont_output = (
                    pitch_output.to(self.device),
                    cont_output.to(self.device),
                )
                self.optimize.zero_grad()
                pitch_logit, cont_logit, _ = self.model(pitch_input, cont_input)

                pitch_loss = self.cross_entropy(
                    pitch_logit.reshape(-1, self.num_pitches), pitch_output.reshape(-1)
                )
                cont_loss = self.mse_error(cont_logit, cont_output)

                loss = pitch_loss + cont_loss

                loss.backward()
                self.optimize.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)
            print(f"epoch {epoch:3d}/{self.epochs}  avg loss {avg_loss:.4f}")
        torch.save(self.model.state_dict(), self.save_path)


if __name__ == "__main__":
    print("Hello from Music-Generator!")
    training = Train(max_files=16)
    training.train()
