from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.data.preprocessing import (
    load_and_prepare_data,
    create_sequences_with_features,
    split_by_vessel,
    normalize_data,
)
from src.models import TrajectoryDataset, EncoderDecoderGRU

DATA_DIR = Path("data")
MODEL_PATH = "best_model_encoder_decoder.pt"
MODEL = EncoderDecoderGRU
INPUT_HOURS = 2
OUTPUT_HOURS = 1
SAMPLING_RATE = 5
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(u"Using device: ", DEVICE)

# Initialize Weights & Biases
wandb.init(
    project="maritime-trajectory-prediction",
    config={
        "input_hours": INPUT_HOURS,
        "output_hours": OUTPUT_HOURS,
        "sampling_rate": SAMPLING_RATE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dropout": 0.3,
        "weight_decay": 1e-5,
        "early_stop_patience": 20,
        "model": "EncoderDecoderGRU",
        "device": str(DEVICE),
    }
)

# Prepare data

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

# Training

model = MODEL(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_seq_len=output_timesteps,
    dropout=0.3,
).to(DEVICE)
print(f"\nModel architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Log model info to wandb
wandb.config.update({"input_size": input_size, "total_parameters": total_params})
wandb.watch(model, log="all", log_freq=100)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

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
    current_lr = optimizer.param_groups[0]['lr']

    # Log metrics to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": current_lr,
    })

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
        
        # Log best model to wandb
        wandb.log({"best_val_loss": best_val_loss})
        wandb.save("best_model_encoder_decoder.pt")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

# Log final summary and finish wandb run
wandb.summary["best_val_loss"] = best_val_loss
wandb.summary["total_epochs_trained"] = epoch + 1
wandb.finish()