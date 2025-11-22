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


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        decoder_hidden = decoder_hidden[-1].unsqueeze(1)

        encoder_transform = self.W_h(encoder_outputs)
        decoder_transform = self.W_s(decoder_hidden)

        energy = torch.tanh(encoder_transform + decoder_transform)
        attention_weights = F.softmax(self.v(energy), dim=1)

        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)

        return context_vector, attention_weights


class EncoderDecoderGRUWithAttention(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_seq_len, dropout=0.3):
        super(EncoderDecoderGRUWithAttention, self).__init__()
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

        self.attention = Attention(hidden_size)

        self.decoder = nn.GRU(
            input_size=2 + hidden_size,
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

        encoder_outputs, hidden = self.encoder(x)

        if torch.isnan(hidden).any() or torch.isnan(encoder_outputs).any():
            print(f"NaN in encoder output!")
            print(f"  Input x range: [{x.min():.2f}, {x.max():.2f}]")
            raise ValueError("NaN in encoder hidden")

        hidden = torch.clamp(hidden, -20.0, 20.0)

        outputs = []
        attention_weights_list = []

        decoder_input = x[:, -1, :2].unsqueeze(1)
        decoder_input = torch.clamp(decoder_input, -5.0, 5.0)

        for t in range(self.output_seq_len):
            context_vector, attention_weights = self.attention(encoder_outputs, hidden)

            attention_weights_list.append(attention_weights)

            decoder_input_with_context = torch.cat([decoder_input, context_vector], dim=2)

            decoder_out, hidden = self.decoder(decoder_input_with_context, hidden)

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

