from pathlib import Path
from tqdm import tqdm
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.data.preprocessing import (
    load_and_prepare_data,
    create_sequences,
    split_by_vessel,
    normalize_data,
)
from src.models import TrajectoryDataset, EncoderDecoderGRU, EncoderDecoderGRUWithAttention
from src.utils.model_utils import HaversineLoss, train_model, evaluate_model
from src.utils import set_seed

set_seed(42)

DATA_DIR = Path("data")
MODEL_PATH = "best_model_encoder_decoder_with_attention.pt"
MODEL = EncoderDecoderGRUWithAttention
INPUT_HOURS = 2
OUTPUT_HOURS = 1
SAMPLING_RATE = 5
HIDDEN_SIZE = 256
NUM_LAYERS = 3
BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device: ", DEVICE)

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
        "teacher_forcing_start": 1.0,
        "teacher_forcing_end": 0.2,
        "model": "EncoderDecoderGRU",
        "device": str(DEVICE),
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
)

# Prepare data

df = load_and_prepare_data(DATA_DIR)
sequences, targets, mmsi_labels, feature_cols = create_sequences(df, INPUT_HOURS, OUTPUT_HOURS, SAMPLING_RATE)

X_train, X_val, X_test, y_train, y_val, y_test = split_by_vessel(
    sequences, targets, mmsi_labels, train_ratio=0.7, val_ratio=0.15, random_seed=42
)

X_train, X_val, X_test, y_train, y_val, y_test, input_scaler, output_scaler = normalize_data(
    X_train, X_val, X_test, y_train, y_val, y_test
)

train_dataset = TrajectoryDataset(X_train, y_train)
val_dataset = TrajectoryDataset(X_val, y_val)
test_dataset = TrajectoryDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
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

criterion = HaversineLoss(output_scaler).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)


print(f"\nStarting training for {EPOCHS} epochs...")
train_losses = []
val_losses = []
best_val_loss = float("inf")
patience_counter = 0
early_stop_patience = 20

for epoch in range(EPOCHS):
    teacher_forcing_ratio = max(0.2, 1.0 - (0.8 * (epoch / EPOCHS)))

    train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE, epoch, EPOCHS, teacher_forcing_ratio)
    val_loss = evaluate_model(model, val_loader, criterion, output_scaler, DEVICE)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    wandb.log(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "teacher_forcing_ratio": teacher_forcing_ratio,
        }
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] - "
        f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
        f"Teacher Forcing: {teacher_forcing_ratio:.3f}, LR: {current_lr:.6f}"
    )

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
            MODEL_PATH,
        )
        print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

        # Log best model to wandb
        wandb.log({"best_val_loss": best_val_loss})
        wandb.save(MODEL_PATH)
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")

print("\n" + "=" * 50)
print("FINAL EVALUATION ON TEST SET")
print("=" * 50)

checkpoint = torch.load(MODEL_PATH, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
test_loss = evaluate_model(model, test_loader, criterion, output_scaler, DEVICE)
print(f"Final Test Loss: {test_loss:.6f}")

wandb.summary["best_val_loss"] = best_val_loss
wandb.summary["test_loss"] = test_loss
wandb.summary["total_epochs_trained"] = epoch + 1

wandb.log(
    {
        "final/test_loss": test_loss,
        "final/val_loss": best_val_loss,
        "final/train_loss": train_losses[-1] if train_losses else 0,
    }
)

print(f"\nFinal Results:")
print(f"  Best Validation Loss: {best_val_loss:.6f}")
print(f"  Final Test Loss: {test_loss:.6f}")
print(f"  Total Epochs: {epoch + 1}")

wandb.finish()
