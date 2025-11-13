import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from preprocessing import (
    load_and_prepare_data,
    create_sequences,
    split_by_vessel,
    normalize_data,
)

# Configuration
DATA_DIR = Path("data")
INPUT_HOURS = 2
OUTPUT_HOURS = 1
SAMPLING_RATE = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
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


class GRUTrajectoryPredictor(nn.Module):
    """GRU model for trajectory prediction."""

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUTrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        gru_out, hidden = self.gru(x)

        # Use the last output of the sequence
        last_output = gru_out[:, -1, :]

        # Predict the output sequence
        output = self.fc(last_output)

        return output


def train_model(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
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

            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(test_loader)


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History (Polars)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history_polars.png", dpi=300, bbox_inches="tight")
    print("\nSaved training history to training_history_polars.png")


def visualize_predictions(model, test_loader, output_scaler, device, n_samples=5):
    """Visualize some predictions."""
    model.eval()
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    with torch.no_grad():
        sequences, targets = next(iter(test_loader))
        sequences = sequences[:n_samples].to(device)
        targets = targets[:n_samples].cpu().numpy()

        predictions = model(sequences).cpu().numpy()

    # Inverse transform
    targets = output_scaler.inverse_transform(targets)
    predictions = output_scaler.inverse_transform(predictions)

    output_timesteps = len(targets[0]) // 2

    for i, ax in enumerate(axes):
        # Reshape back to (timesteps, 2)
        true_traj = targets[i].reshape(output_timesteps, 2)
        pred_traj = predictions[i].reshape(output_timesteps, 2)

        ax.plot(true_traj[:, 1], true_traj[:, 0], "b-o", label="True", markersize=4)
        ax.plot(pred_traj[:, 1], pred_traj[:, 0], "r-o", label="Predicted", markersize=4)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Trajectory Prediction {i+1}")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("predictions_polars.png", dpi=300, bbox_inches="tight")
    print("Saved predictions to predictions_polars.png")


def main():
    print(f"Using device: {DEVICE}")
    print("Using Polars for data loading\n")

    df = load_and_prepare_data(DATA_DIR)

    # Create sequences with MMSI tracking
    sequences, targets, mmsi_labels, feature_cols = create_sequences(df, INPUT_HOURS, OUTPUT_HOURS, SAMPLING_RATE)

    # Split by vessel (no vessel appears in both train and test)
    X_train, X_test, y_train, y_test = split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.8)

    # Normalize
    X_train, X_test, y_train, y_test, input_scaler, output_scaler = normalize_data(X_train, X_test, y_train, y_test)

    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    input_size = len(feature_cols)
    output_size = y_train.shape[1]  # Flattened lat/lon sequence

    model = GRUTrajectoryPredictor(
        input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=output_size
    ).to(DEVICE)

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    print("Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")

    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

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
                    },
                },
                "best_model_polars.pt",
            )
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

    # Plot training history
    plot_training_history(train_losses, val_losses)

    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, output_scaler, DEVICE)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: best_model_polars.pt")
    print(f"Training history saved to: training_history_polars.png")
    print(f"Predictions saved to: predictions_polars.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
