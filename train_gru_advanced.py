import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from preprocessing import (
    load_and_prepare_data,
    create_sequences_with_features,
    split_by_vessel,
    normalize_data,
)

# Configuration
DATA_DIR = Path("data")
INPUT_HOURS = 2
OUTPUT_HOURS = 1
SAMPLING_RATE = 5  # Sample every N minutes
HIDDEN_SIZE = 256  # Increased
NUM_LAYERS = 3  # Increased
BATCH_SIZE = 64  # Increased
EPOCHS = 100  # Increased
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrajectoryDataset(Dataset):
    """Dataset for vessel trajectory sequences."""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class AttentionLayer(nn.Module):
    """Attention mechanism for focusing on relevant timesteps."""

    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, gru_output):
        # gru_output shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(gru_output), dim=1)
        # attention_weights shape: (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(attention_weights * gru_output, dim=1)
        # context shape: (batch, hidden_size)

        return context, attention_weights


class AdvancedGRUTrajectoryPredictor(nn.Module):
    """Advanced GRU model with attention and residual connections."""

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
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)
        x = F.relu(x)

        # GRU layer
        gru_out, hidden = self.gru(x)
        # gru_out shape: (batch, seq_len, hidden_size * 2)

        # Attention
        context, attention_weights = self.attention(gru_out)
        # context shape: (batch, hidden_size * 2)

        # Fully connected layers with residual connections
        x = F.relu(self.layer_norm1(self.fc1(context)))
        x = self.dropout(x)

        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)

        output = self.fc3(x)

        return output, attention_weights


class EncoderDecoderGRU(nn.Module):
    """Encoder-Decoder architecture for sequence-to-sequence prediction."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, output_seq_len):
        super(EncoderDecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_seq_len = output_seq_len

        # Encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )

        # Decoder
        self.decoder = nn.GRU(
            input_size=2,  # lat, lon
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 2)  # Output lat, lon

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.5):
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Encode
        encoder_out, hidden = self.encoder(x)

        # Decode
        outputs = []

        # Initial decoder input (last position from encoder)
        decoder_input = x[:, -1, :2].unsqueeze(1)  # (batch, 1, 2)

        for t in range(self.output_seq_len):
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc(decoder_out)
            outputs.append(prediction)

            # Teacher forcing
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t : t + 1, :]
            else:
                decoder_input = prediction

        # Concatenate outputs
        outputs = torch.cat(outputs, dim=1)  # (batch, output_seq_len, 2)
        outputs = outputs.reshape(batch_size, -1)  # Flatten

        return outputs


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Handle different model outputs
        if isinstance(model, AdvancedGRUTrajectoryPredictor):
            outputs, _ = model(sequences)
        else:
            outputs = model(sequences)

        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            if isinstance(model, AdvancedGRUTrajectoryPredictor):
                outputs, _ = model(sequences)
            else:
                outputs = model(sequences)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training History (Advanced Model)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (log scale)", fontsize=12)
    plt.title("Training History - Log Scale", fontsize=14)
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history_advanced.png", dpi=300, bbox_inches="tight")
    print("\nSaved training history to training_history_advanced.png")


def visualize_predictions(model, test_loader, output_scaler, device, n_samples=5):
    """Visualize predictions with attention weights."""
    model.eval()
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    with torch.no_grad():
        sequences, targets = next(iter(test_loader))
        sequences_plot = sequences[:n_samples].to(device)
        targets = targets[:n_samples].cpu().numpy()

        if isinstance(model, AdvancedGRUTrajectoryPredictor):
            predictions, attention_weights = model(sequences_plot)
            predictions = predictions.cpu().numpy()
            attention_weights = attention_weights.cpu().numpy()
        else:
            predictions = model(sequences_plot).cpu().numpy()
            attention_weights = None

    targets = output_scaler.inverse_transform(targets)
    predictions = output_scaler.inverse_transform(predictions)

    output_timesteps = len(targets[0]) // 2

    for i in range(n_samples):
        true_traj = targets[i].reshape(output_timesteps, 2)
        pred_traj = predictions[i].reshape(output_timesteps, 2)

        # Plot trajectories
        axes[i, 0].plot(true_traj[:, 1], true_traj[:, 0], "b-o", label="True", markersize=6, linewidth=2)
        axes[i, 0].plot(pred_traj[:, 1], pred_traj[:, 0], "r-o", label="Predicted", markersize=6, linewidth=2)
        axes[i, 0].set_xlabel("Longitude", fontsize=11)
        axes[i, 0].set_ylabel("Latitude", fontsize=11)
        axes[i, 0].set_title(f"Trajectory Prediction {i+1}", fontsize=12)
        axes[i, 0].legend(fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        # Plot attention weights if available
        if attention_weights is not None:
            axes[i, 1].plot(attention_weights[i].squeeze())
            axes[i, 1].set_xlabel("Time Step", fontsize=11)
            axes[i, 1].set_ylabel("Attention Weight", fontsize=11)
            axes[i, 1].set_title(f"Attention Weights {i+1}", fontsize=12)
            axes[i, 1].grid(True, alpha=0.3)
        else:
            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("predictions_advanced.png", dpi=300, bbox_inches="tight")
    print("Saved predictions to predictions_advanced.png")


def main():
    print("=" * 70)
    print("ADVANCED GRU TRAJECTORY PREDICTOR")
    print("=" * 70)
    print(f"Using device: {DEVICE}")
    print("Using Polars for data loading\n")

    df = load_and_prepare_data(DATA_DIR)

    # Create sequences with enhanced features and MMSI tracking
    sequences, targets, mmsi_labels, feature_cols = create_sequences_with_features(
        df, INPUT_HOURS, OUTPUT_HOURS, SAMPLING_RATE
    )

    # Split by vessel (no vessel appears in both train and test)
    X_train, X_test, y_train, y_test = split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.8)

    # Normalize
    X_train, X_test, y_train, y_test, input_scaler, output_scaler = normalize_data(X_train, X_test, y_train, y_test)

    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize advanced model
    input_size = len(feature_cols)
    output_size = y_train.shape[1]

    model = AdvancedGRUTrajectoryPredictor(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=output_size, dropout=0.3
    ).to(DEVICE)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer with weight decay
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    print("Using ReduceLROnPlateau scheduler (factor=0.5, patience=10)")
    print("Using AdamW optimizer with weight decay=1e-5")

    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate_model(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - " f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "input_scaler": input_scaler,
                    "output_scaler": output_scaler,
                    "config": {
                        "input_size": input_size,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "output_size": output_size,
                        "input_hours": INPUT_HOURS,
                        "output_hours": OUTPUT_HOURS,
                        "sampling_rate": SAMPLING_RATE,
                        "feature_cols": feature_cols,
                    },
                },
                "best_model_advanced.pt",
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, output_scaler, DEVICE)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Model saved to: best_model_advanced.pt")
    print(f"Training history saved to: training_history_advanced.png")
    print(f"Predictions saved to: predictions_advanced.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
