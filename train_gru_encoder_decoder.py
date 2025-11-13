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

DATA_DIR = Path("data")
INPUT_HOURS = 2
OUTPUT_HOURS = 1
SAMPLING_RATE = 5
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 64
EPOCHS = 100
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


class EncoderDecoderGRU(nn.Module):
    """Encoder-Decoder architecture for sequence-to-sequence prediction."""

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
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        encoder_out, hidden = self.encoder(x)

        outputs = []

        decoder_input = x[:, -1, :2].unsqueeze(1)

        for t in range(self.output_seq_len):
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            prediction = self.fc(decoder_out)
            outputs.append(prediction)

            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = target_seq[:, t : t + 1, :]
            else:
                decoder_input = prediction

        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.reshape(batch_size, -1)

        return outputs


def train_model(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    teacher_forcing_ratio = max(0.5 * (1 - epoch / total_epochs), 0.0)

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output_timesteps = targets.shape[1] // 2
        target_seq = targets.reshape(targets.shape[0], output_timesteps, 2)

        outputs = model(sequences, target_seq, teacher_forcing_ratio)

        loss = criterion(outputs, targets)
        loss.backward()

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

            outputs = model(sequences, target_seq=None, teacher_forcing_ratio=0.0)

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
    plt.title("Training History (Encoder-Decoder)", fontsize=14)
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
    plt.savefig("training_history_encoder_decoder.png", dpi=300, bbox_inches="tight")
    print("\nSaved training history to training_history_encoder_decoder.png")


def visualize_predictions(model, test_loader, output_scaler, device, n_samples=5):
    """Visualize predictions."""
    model.eval()
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    with torch.no_grad():
        sequences, targets = next(iter(test_loader))
        sequences_plot = sequences[:n_samples].to(device)
        targets = targets[:n_samples].cpu().numpy()

        predictions = model(sequences_plot, target_seq=None, teacher_forcing_ratio=0.0)
        predictions = predictions.cpu().numpy()

    targets = output_scaler.inverse_transform(targets)
    predictions = output_scaler.inverse_transform(predictions)

    output_timesteps = len(targets[0]) // 2

    for i in range(n_samples):
        true_traj = targets[i].reshape(output_timesteps, 2)
        pred_traj = predictions[i].reshape(output_timesteps, 2)

        axes[i, 0].plot(true_traj[:, 1], true_traj[:, 0], "b-o", label="True", markersize=6, linewidth=2)
        axes[i, 0].plot(pred_traj[:, 1], pred_traj[:, 0], "r-o", label="Predicted", markersize=6, linewidth=2)
        axes[i, 0].set_xlabel("Longitude", fontsize=11)
        axes[i, 0].set_ylabel("Latitude", fontsize=11)
        axes[i, 0].set_title(f"Trajectory Prediction {i+1}", fontsize=12)
        axes[i, 0].legend(fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        error = np.linalg.norm(true_traj - pred_traj, axis=1)
        axes[i, 1].plot(error, "r-o", linewidth=2)
        axes[i, 1].set_xlabel("Timestep", fontsize=11)
        axes[i, 1].set_ylabel("Position Error (degrees)", fontsize=11)
        axes[i, 1].set_title(f"Prediction Error {i+1}", fontsize=12)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions_encoder_decoder.png", dpi=300, bbox_inches="tight")
    print("Saved predictions to predictions_encoder_decoder.png")


def main():
    print("=" * 70)
    print("ENCODER-DECODER GRU TRAJECTORY PREDICTOR")
    print("=" * 70)
    print(f"Using device: {DEVICE}")
    print("Using Polars for data loading\n")

    df = load_and_prepare_data(DATA_DIR)

    sequences, targets, mmsi_labels, feature_cols = create_sequences_with_features(
        df, INPUT_HOURS, OUTPUT_HOURS, SAMPLING_RATE
    )

    X_train, X_test, y_train, y_test = split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.8)

    X_train, X_test, y_train, y_test, input_scaler, output_scaler = normalize_data(X_train, X_test, y_train, y_test)

    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    input_size = len(feature_cols)
    output_timesteps = y_train.shape[1] // 2

    model = EncoderDecoderGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_seq_len=output_timesteps,
        dropout=0.3,
    ).to(DEVICE)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    print("Using ReduceLROnPlateau scheduler (factor=0.5, patience=10)")
    print("Using AdamW optimizer with weight decay=1e-5")
    print("Using scheduled teacher forcing (0.5 -> 0.0)")

    print(f"\nStarting training for {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS)
        val_loss = evaluate_model(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - " f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

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
                        "output_seq_len": output_timesteps,
                        "input_hours": INPUT_HOURS,
                        "output_hours": OUTPUT_HOURS,
                        "sampling_rate": SAMPLING_RATE,
                        "feature_cols": feature_cols,
                    },
                },
                "best_model_encoder_decoder.pt",
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

    plot_training_history(train_losses, val_losses)

    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, output_scaler, DEVICE)

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Model saved to: best_model_encoder_decoder.pt")
    print(f"Training history saved to: training_history_encoder_decoder.png")
    print(f"Predictions saved to: predictions_encoder_decoder.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

