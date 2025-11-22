import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class AttentionLayer(nn.Module):

    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, gru_output):
        attention_weights = F.softmax(self.attention(gru_output), dim=1)
        context = torch.sum(attention_weights * gru_output, dim=1)
        return context, attention_weights


class AdvancedGRUTrajectoryPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(AdvancedGRUTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional

        # Fully connected layers with residual connection
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)

    def forward(self, x):
        x = self.input_projection(x)
        x = F.relu(x)

        # GRU layer
        gru_out, hidden = self.gru(x)

        # Attention
        context, attention_weights = self.attention(gru_out)

        # Fully connected layers with residual connections
        x = F.relu(self.layer_norm1(self.fc1(context)))
        x = self.dropout(x)

        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)

        output = self.fc3(x)

        return output, attention_weights


class EncoderDecoderGRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_seq_len, dropout=0.3):
        super(EncoderDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len

        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.decoder = nn.GRU(
            input_size=2,  # lat, lon
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, 2)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
        nn.init.zeros_(self.fc.bias)
    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        x = torch.clamp(x, -10.0, 10.0)

        encoder_out, hidden = self.encoder(x)

        if torch.isnan(hidden).any() or torch.isnan(encoder_out).any():
            print(f"NaN in encoder output!")
            print(f"  Input x range: [{x.min():.2f}, {x.max():.2f}]")
            raise ValueError("NaN in encoder hidden")
        
        hidden = torch.clamp(hidden, -20.0, 20.0)

        outputs = []

        decoder_input = x[:, -1, :2].unsqueeze(1)
        decoder_input = torch.clamp(decoder_input, -5.0, 5.0)

        for t in range(self.output_seq_len):
            decoder_out, hidden = self.decoder(decoder_input, hidden)

            if torch.isnan(decoder_out).any() or torch.isnan(hidden).any():
                print(f"NaN in decoder at timestep {t}")
                print(f"  decoder_input range: [{decoder_input.min():.2f}, {decoder_input.max():.2f}]")
                print(f"  decoder_out has NaN: {torch.isnan(decoder_out).any()}")
                print(f"  hidden has NaN: {torch.isnan(hidden).any()}")
                raise ValueError(f"NaN in decoder at timestep {t}")

            hidden = torch.clamp(hidden, -20.0, 20.0)

            delta = self.fc(decoder_out)
            delta = torch.tanh(delta)

            prediction = decoder_input + delta
            prediction = torch.clamp(prediction, -5.0, 5.0)

            outputs.append(prediction)

            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t : t + 1, :].detach()
                decoder_input = torch.clamp(decoder_input, -5.0, 5.0)
            else:
                decoder_input = prediction

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.reshape(batch_size, -1)

        return outputs


class SimpleGRUPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(SimpleGRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, hidden = self.gru(x)

        # Use last hidden state
        output = self.fc(hidden[-1])

        return output
