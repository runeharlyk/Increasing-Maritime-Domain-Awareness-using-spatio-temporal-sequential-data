import numpy as np
import polars as pl
from pathlib import Path
import torch
import matplotlib.pyplot as plt

from src.utils.geo import haversine_distance_torch, haversine_distance
from src.utils import config
from src.data.preprocessing import check_sequence_distance


class HaversineLoss(torch.nn.Module):
    def __init__(self, output_scaler):
        super(HaversineLoss, self).__init__()
        self.register_buffer("scale_", torch.tensor(output_scaler.scale_, dtype=torch.float32))
        self.register_buffer("mean_", torch.tensor(output_scaler.mean_, dtype=torch.float32))

    def forward(self, predictions, targets):
        predictions = torch.clamp(predictions, -10.0, 10.0)
        targets = torch.clamp(targets, -10.0, 10.0)

        pred_reshaped = predictions.reshape(-1, 2)
        target_reshaped = targets.reshape(-1, 2)

        pred_unscaled = pred_reshaped * self.scale_ + self.mean_
        target_unscaled = target_reshaped * self.scale_ + self.mean_

        pred_lat = torch.clamp(pred_unscaled[:, 0], -90.0, 90.0)
        pred_lon = torch.clamp(pred_unscaled[:, 1], -180.0, 180.0)
        target_lat = torch.clamp(target_unscaled[:, 0], -90.0, 90.0)
        target_lon = torch.clamp(target_unscaled[:, 1], -180.0, 180.0)

        distances = haversine_distance_torch(pred_lat, pred_lon, target_lat, target_lon)

        if torch.isnan(distances).any():
            print(f"NaN detected in distances!")
            print(f"  pred_lat range: [{pred_lat.min():.2f}, {pred_lat.max():.2f}]")
            print(f"  pred_lon range: [{pred_lon.min():.2f}, {pred_lon.max():.2f}]")
            print(f"  target_lat range: [{target_lat.min():.2f}, {target_lat.max():.2f}]")
            print(f"  target_lon range: [{target_lon.min():.2f}, {target_lon.max():.2f}]")
            print(f"  pred_unscaled (before clamp) range: [{pred_unscaled.min():.2f}, {pred_unscaled.max():.2f}]")
            print(f"  predictions (normalized) range: [{predictions.min():.2f}, {predictions.max():.2f}]")

        return distances.mean()


def train_model(model, train_loader, criterion, optimizer, device, epoch, total_epochs, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0

    for batch_idx, (sequences, targets) in enumerate(train_loader):
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output_timesteps = targets.shape[1] // 2
        target_seq = targets.reshape(targets.shape[0], output_timesteps, 2)

        outputs = model(sequences, target_seq, teacher_forcing_ratio)

        if torch.isnan(outputs).any():
            print(f"NaN detected in model outputs at epoch {epoch+1}, batch {batch_idx}")
            print(f"  outputs range: [{outputs.min():.2f}, {outputs.max():.2f}]")
            print(f"  sequences range: [{sequences.min():.2f}, {sequences.max():.2f}]")
            print(f"  Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
            raise ValueError("NaN in model outputs")

        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            print(f"NaN loss at epoch {epoch+1}, batch {batch_idx}")
            raise ValueError("NaN loss detected")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


    """Evaluate the model."""
def evaluate_model(model, test_loader, criterion, output_scaler, device, return_predictions=False):
    model.eval()
    total_loss = 0
    
    if return_predictions:
        predictions = []
        true_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            if return_predictions:
                true_targets.append(targets.cpu().numpy())
            
            sequences = sequences.to(device)
            targets = targets.to(device)

            outputs = model(sequences, target_seq=None, teacher_forcing_ratio=0.0)
            
            if return_predictions:
                predictions.append(outputs.cpu().numpy())
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    
    if return_predictions:
        predictions = np.vstack(predictions)
        predictions_reshaped = predictions.reshape(-1, 2)
        predictions = output_scaler.inverse_transform(predictions_reshaped).reshape(predictions.shape[0], -1)

        true_targets = np.vstack(true_targets)
        true_targets_reshaped = true_targets.reshape(-1, 2)
        true_targets = output_scaler.inverse_transform(true_targets_reshaped).reshape(true_targets.shape[0], -1)
        
        return avg_loss, predictions, true_targets
    
    return avg_loss


def load_model_and_config(model_path, model_class, device="cpu"):
    print(f"Loading model from {model_path}...")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)

    config = checkpoint["config"]
    input_scaler = checkpoint["input_scaler"]
    output_scaler = checkpoint["output_scaler"]

    if "output_seq_len" in config:
        model = model_class(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_seq_len=config["output_seq_len"],
            dropout=config.get("dropout", 0.3),
        )
    else:
        model = model_class(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=config["output_size"],
            dropout=config.get("dropout", 0.3),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    print(f"Model loaded successfully!")
    print(f"  Input hours: {config['input_hours']}")
    print(f"  Output hours: {config['output_hours']}")
    print(f"  Sampling rate: {config['sampling_rate']} minutes")

    return model, config, input_scaler, output_scaler



def create_prediction_sequences(df, config_dict, n_vessels=None):
    input_hours = config_dict["input_hours"]
    output_hours = config_dict["output_hours"]
    sampling_rate = config_dict["sampling_rate"]
    feature_cols = config_dict["feature_cols"]

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    print(f"\nCreating prediction sequences:")
    print(f"  Input: {input_hours}h ({input_timesteps} timesteps)")
    print(f"  Output: {output_hours}h ({output_timesteps} timesteps)")
    print(f"  Sampling rate: {sampling_rate} minutes")
    total_hours = input_hours + output_hours
    print(
        f"  Filtering sequences: min {config.MIN_DISTANCE_KM} km/h ({config.MIN_DISTANCE_KM * total_hours} km over {total_hours}h)"
    )

    base_features = ["Latitude", "Longitude", "SOG", "COG"]

    agg_cols = [
        pl.col("Latitude").first(),
        pl.col("Longitude").first(),
        pl.col("SOG").first(),
        pl.col("COG").first(),
        pl.col("GlobalSegment").first(),
    ]

    if "FileIndex" in df.columns:
        agg_cols.append(pl.col("FileIndex").first())

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", group_by="MMSI")
        .agg(agg_cols)
        .drop_nulls(subset=base_features)
    )

    if any(
        col in feature_cols
        for col in [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "SOG_diff",
            "COG_diff_sin",
            "COG_diff_cos",
        ]
    ):
        df_processed = (
            df_processed.with_columns(
                [
                    (2 * np.pi * pl.col("Timestamp").dt.hour() / 24.0).sin().alias("hour_sin"),
                    (2 * np.pi * pl.col("Timestamp").dt.hour() / 24.0).cos().alias("hour_cos"),
                    (2 * np.pi * pl.col("Timestamp").dt.weekday() / 7.0).sin().alias("day_of_week_sin"),
                    (2 * np.pi * pl.col("Timestamp").dt.weekday() / 7.0).cos().alias("day_of_week_cos"),
                ]
            )
            .with_columns(
                [
                    pl.col("SOG").diff().over(["MMSI", "GlobalSegment"]).fill_null(0).alias("SOG_diff"),
                    (((pl.col("COG").diff().over(["MMSI", "GlobalSegment"]).fill_null(0) + 180) % 360) - 180).alias(
                        "COG_diff_raw"
                    ),
                ]
            )
            .with_columns(
                [
                    (pl.col("COG_diff_raw") * np.pi / 180.0).sin().alias("COG_diff_sin"),
                    (pl.col("COG_diff_raw") * np.pi / 180.0).cos().alias("COG_diff_cos"),
                ]
            )
        )

    if any(col in feature_cols for col in ["COG_sin", "COG_cos"]):
        df_processed = df_processed.with_columns(
            [
                (pl.col("COG") * np.pi / 180.0).sin().alias("COG_sin"),
                (pl.col("COG") * np.pi / 180.0).cos().alias("COG_cos"),
            ]
        )

    group_cols = ["MMSI", "GlobalSegment"]
    print(f"  Processing by {group_cols}")

    segment_groups = df_processed.partition_by(group_cols, as_dict=True)

    sequences = []
    targets = []
    mmsi_list = []
    full_trajectories = []
    timestamps_list = []

    count = 0
    skipped_irregular = 0
    skipped_stationary = 0
    skipped_harbor_approach = 0
    skipped_other = 0
    
    for group_key, vessel_data in segment_groups.items():
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(feature_cols).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()
        timestamps = vessel_data.select("Timestamp").to_numpy().flatten()

        mmsi = group_key[0] if isinstance(group_key, tuple) else group_key

        input_seq = data_array[0:input_timesteps]
        output_seq = lat_lon[input_timesteps : input_timesteps + output_timesteps]
        full_traj = lat_lon[0 : input_timesteps + output_timesteps]
        traj_timestamps = timestamps[0 : input_timesteps + output_timesteps]

        if np.isnan(input_seq).any() or np.isnan(output_seq).any():
            continue

        time_diffs = np.diff(traj_timestamps).astype("timedelta64[m]").astype(int)
        expected_diff = sampling_rate
        if not np.all(np.abs(time_diffs - expected_diff) <= 1):
            skipped_irregular += 1
            continue

        is_valid, reason = check_sequence_distance(full_traj, traj_timestamps, input_timesteps)
        if not is_valid:
            if reason == "insufficient_output_distance":
                skipped_harbor_approach += 1
            elif reason == "insufficient_total_distance":
                skipped_stationary += 1
            else:
                skipped_other += 1
            continue

        sequences.append(input_seq)
        targets.append(output_seq.flatten())
        mmsi_list.append(mmsi)
        full_trajectories.append(full_traj)
        timestamps_list.append(traj_timestamps)
        count += 1

        if n_vessels is not None and count >= n_vessels:
            break

    if skipped_irregular > 0:
        print(f"  Skipped {skipped_irregular} sequences with irregular time spacing")
    if skipped_stationary > 0:
        print(f"  Skipped {skipped_stationary} sequences with insufficient distance traveled (total)")
    if skipped_harbor_approach > 0:
        print(f"  Skipped {skipped_harbor_approach} sequences with slow/stopping motion in prediction window (likely harbor approaches)")
    if skipped_other > 0:
        print(f"  Skipped {skipped_other} sequences for other reasons")

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"Created {len(sequences)} sequences from {len(mmsi_list)} vessels")

    return sequences, targets, mmsi_list, full_trajectories, timestamps_list


def predict_trajectories(model, sequences, input_scaler, output_scaler, device="cpu"):
    print("\nMaking predictions...")

    n_samples, n_timesteps, n_features = sequences.shape
    sequences_scaled = sequences.copy()

    feature_cols = config.FEATURE_COLS
    cols_to_norm = config.COLS_TO_NORMALIZE
    features_to_normalize = [i for i, col in enumerate(feature_cols) if col in cols_to_norm]

    sequences_norm = sequences[:, :, features_to_normalize].reshape(-1, len(features_to_normalize))
    sequences_scaled[:, :, features_to_normalize] = input_scaler.transform(sequences_norm).reshape(
        n_samples, n_timesteps, len(features_to_normalize)
    )

    sequences_tensor = torch.FloatTensor(sequences_scaled).to(device)

    with torch.no_grad():
        model_output = model(sequences_tensor)

        if isinstance(model_output, tuple):
            predictions, attention_weights = model_output
            predictions = predictions.cpu().numpy()
            attention_weights = attention_weights.cpu().numpy()
        else:
            predictions = model_output.cpu().numpy()
            attention_weights = None

    predictions_reshaped = predictions.reshape(-1, 2)
    predictions = output_scaler.inverse_transform(predictions_reshaped).reshape(predictions.shape[0], -1)

    return predictions, attention_weights


def plot_training_history(train_losses, val_losses, model_name="encoder_decoder"):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training History ({model_name})", fontsize=14)
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
    filename = f"training_history_{model_name}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nSaved training history to {filename}")


def visualize_predictions(
    model, test_loader, output_scaler, device, n_samples=5, model_name="encoder_decoder", save_fig=False
):
    model.eval()
    fig, axes = plt.subplots(n_samples, 2, figsize=(16, 4 * n_samples))

    with torch.no_grad():
        sequences, targets = next(iter(test_loader))
        sequences_plot = sequences[:n_samples].to(device)
        targets = targets[:n_samples].cpu().numpy()

        predictions = model(sequences_plot, target_seq=None, teacher_forcing_ratio=0.0)
        predictions = predictions.cpu().numpy()

    output_timesteps = targets.shape[1] // 2

    targets_reshaped = targets.reshape(-1, 2)
    targets = output_scaler.inverse_transform(targets_reshaped).reshape(n_samples, -1)

    predictions_reshaped = predictions.reshape(-1, 2)
    predictions = output_scaler.inverse_transform(predictions_reshaped).reshape(n_samples, -1)

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
    if save_fig:
        plt.savefig(f"predictions_{model_name}.png", dpi=300, bbox_inches="tight")
        print(f"Saved predictions to predictions_{model_name}.png")
