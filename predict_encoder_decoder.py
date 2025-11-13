import numpy as np
import polars as pl
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import folium
from folium import plugins
import random

DATA_DIR = Path("parquet")
MODEL_PATH = "best_model_encoder_decoder.pt"
PARQUET_FILES = ["aisdk-2024-03-21.parquet", "aisdk-2024-03-22.parquet"]
OUTPUT_HTML = "prediction_map_encoder_decoder.html"
N_VESSELS = 5


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

    def forward(self, x, target_seq=None, teacher_forcing_ratio=0.0):
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


def load_model_and_config(model_path):
    """Load trained model and configuration."""
    print(f"Loading model from {model_path}...")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    config = checkpoint["config"]
    input_scaler = checkpoint["input_scaler"]
    output_scaler = checkpoint["output_scaler"]

    model = EncoderDecoderGRU(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_seq_len=config["output_seq_len"],
        dropout=0.3,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Input hours: {config['input_hours']}")
    print(f"  Output hours: {config['output_hours']}")
    print(f"  Sampling rate: {config['sampling_rate']} minutes")

    return model, config, input_scaler, output_scaler


def load_trajectory_data(data_dir, parquet_files, n_vessels=5):
    """Load 3-hour trajectory data from parquet files."""
    print(f"\nLoading trajectory data from {len(parquet_files)} file(s)...")

    parquet_paths = [data_dir / f for f in parquet_files]

    for parquet_path in parquet_paths:
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    dfs = []
    for parquet_path in parquet_paths:
        print(f"  Loading {parquet_path.name}...")
        df_temp = pl.read_parquet(parquet_path)
        dfs.append(df_temp)

    df = pl.concat(dfs, how="vertical")

    print(f"Total rows: {len(df):,}")
    print(f"Unique vessels (MMSI): {df['MMSI'].n_unique()}")

    df = df.sort(["MMSI", "Timestamp"])

    if df["Timestamp"].dtype != pl.Datetime:
        df = df.with_columns(pl.col("Timestamp").cast(pl.Datetime))

    return df


def create_3hour_sequences(df, config, n_vessels=5):
    """Create 3-hour sequences: 2h input + 1h output."""
    input_hours = config["input_hours"]
    output_hours = config["output_hours"]
    sampling_rate = config["sampling_rate"]
    feature_cols = config["feature_cols"]

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    print(f"\nCreating 3-hour sequences:")
    print(f"  Input: {input_hours}h ({input_timesteps} timesteps)")
    print(f"  Output: {output_hours}h ({output_timesteps} timesteps)")
    print(f"  Sampling rate: {sampling_rate} minutes")

    base_features = ["Latitude", "Longitude", "SOG", "COG"]

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", group_by="MMSI")
        .agg(
            [
                pl.col("Latitude").first(),
                pl.col("Longitude").first(),
                pl.col("SOG").first(),
                pl.col("COG").first(),
            ]
        )
        .drop_nulls(subset=base_features)
        .with_columns(
            [
                (pl.col("Timestamp").dt.hour() / 24.0).alias("hour"),
                (pl.col("Timestamp").dt.weekday() / 7.0).alias("day_of_week"),
            ]
        )
        .with_columns(
            [
                pl.col("SOG").diff().over("MMSI").fill_null(0).alias("SOG_diff"),
                pl.col("COG").diff().over("MMSI").fill_null(0).alias("COG_diff"),
            ]
        )
    )

    vessel_groups = df_processed.partition_by("MMSI", as_dict=True)

    sequences = []
    targets = []
    mmsi_list = []
    full_trajectories = []
    timestamps_list = []

    count = 0
    for mmsi, vessel_data in vessel_groups.items():
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(feature_cols).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()
        timestamps = vessel_data.select("Timestamp").to_numpy().flatten()

        input_seq = data_array[0:input_timesteps]
        output_seq = lat_lon[input_timesteps : input_timesteps + output_timesteps]
        full_traj = lat_lon[0 : input_timesteps + output_timesteps]
        traj_timestamps = timestamps[0 : input_timesteps + output_timesteps]

        if not (np.isnan(input_seq).any() or np.isnan(output_seq).any()):
            sequences.append(input_seq)
            targets.append(output_seq.flatten())
            mmsi_list.append(mmsi[0])
            full_trajectories.append(full_traj)
            timestamps_list.append(traj_timestamps)
            count += 1

            if count >= n_vessels:
                break

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"Created {len(sequences)} 3-hour sequences from {len(mmsi_list)} vessels")

    return sequences, targets, mmsi_list, full_trajectories, timestamps_list


def predict_trajectories(model, sequences, input_scaler, output_scaler):
    """Make predictions using the trained model."""
    print("\nMaking predictions...")

    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    sequences_scaled = input_scaler.transform(sequences_reshaped)
    sequences_scaled = sequences_scaled.reshape(n_samples, n_timesteps, n_features)

    sequences_tensor = torch.FloatTensor(sequences_scaled)

    with torch.no_grad():
        predictions = model(sequences_tensor, target_seq=None, teacher_forcing_ratio=0.0)
        predictions = predictions.numpy()

    predictions = output_scaler.inverse_transform(predictions)

    return predictions


def plot_trajectories(full_trajectories, predictions, mmsi_list, output_hours, timestamps_list):
    """Plot actual vs predicted trajectories using matplotlib."""
    n_vessels = len(full_trajectories)
    output_timesteps = predictions.shape[1] // 2

    fig, axes = plt.subplots(n_vessels, 2, figsize=(16, 4 * n_vessels))

    if n_vessels == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_vessels):
        full_traj = full_trajectories[i]
        pred_traj = predictions[i].reshape(output_timesteps, 2)

        input_traj = full_traj[: len(full_traj) - output_timesteps]
        output_traj = full_traj[len(full_traj) - output_timesteps :]

        axes[i, 0].plot(
            input_traj[:, 1],
            input_traj[:, 0],
            "g-o",
            label="Input (2h)",
            markersize=4,
            linewidth=2,
            alpha=0.7,
        )
        axes[i, 0].plot(
            output_traj[:, 1],
            output_traj[:, 0],
            "b-o",
            label="Actual (1h)",
            markersize=6,
            linewidth=2,
        )
        axes[i, 0].plot(
            pred_traj[:, 1],
            pred_traj[:, 0],
            "r--o",
            label="Predicted (1h)",
            markersize=6,
            linewidth=2,
        )

        axes[i, 0].plot(input_traj[0, 1], input_traj[0, 0], "go", markersize=12, label="Start")
        axes[i, 0].plot(output_traj[-1, 1], output_traj[-1, 0], "bs", markersize=12, label="End (Actual)")
        axes[i, 0].plot(pred_traj[-1, 1], pred_traj[-1, 0], "r^", markersize=12, label="End (Predicted)")

        axes[i, 0].set_xlabel("Longitude", fontsize=11)
        axes[i, 0].set_ylabel("Latitude", fontsize=11)
        axes[i, 0].set_title(f"Vessel {mmsi_list[i]} - Trajectory Prediction (Encoder-Decoder)", fontsize=12)
        axes[i, 0].legend(fontsize=9, loc="best")
        axes[i, 0].grid(True, alpha=0.3)

        error = np.linalg.norm(output_traj - pred_traj, axis=1)
        timesteps = np.arange(len(error))

        axes[i, 1].plot(timesteps, error, "r-o", linewidth=2, markersize=6)
        axes[i, 1].set_xlabel("Timestep (5-min intervals)", fontsize=11)
        axes[i, 1].set_ylabel("Position Error (degrees)", fontsize=11)
        axes[i, 1].set_title(f"Vessel {mmsi_list[i]} - Prediction Error", fontsize=12)
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=error.mean(), color="orange", linestyle="--", label=f"Mean: {error.mean():.4f}")
        axes[i, 1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("trajectory_predictions_encoder_decoder.png", dpi=300, bbox_inches="tight")
    print("\nSaved matplotlib plot to trajectory_predictions_encoder_decoder.png")


def create_interactive_map(full_trajectories, predictions, mmsi_list, output_hours):
    """Create interactive folium map with predictions."""
    print("\nCreating interactive map...")

    all_lats = []
    all_lons = []
    for traj in full_trajectories:
        all_lats.extend(traj[:, 0])
        all_lons.extend(traj[:, 1])

    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

    output_timesteps = predictions.shape[1] // 2

    for i, full_traj in enumerate(full_trajectories):
        pred_traj = predictions[i].reshape(output_timesteps, 2)
        mmsi = mmsi_list[i]

        input_traj = full_traj[: len(full_traj) - output_timesteps]
        output_traj = full_traj[len(full_traj) - output_timesteps :]

        color_input = "#{:06x}".format(random.randint(0, 0xFFFFFF))

        input_coords = [[lat, lon] for lat, lon in input_traj]
        folium.PolyLine(
            input_coords,
            color=color_input,
            weight=4,
            opacity=0.8,
            popup=f"MMSI {mmsi}: Input (2h)",
        ).add_to(m)

        actual_coords = [[lat, lon] for lat, lon in output_traj]
        folium.PolyLine(
            actual_coords,
            color="blue",
            weight=3,
            opacity=0.7,
            popup=f"MMSI {mmsi}: Actual (1h)",
        ).add_to(m)

        pred_coords = [[lat, lon] for lat, lon in pred_traj]
        folium.PolyLine(
            pred_coords,
            color="red",
            weight=3,
            opacity=0.7,
            dash_array="10, 5",
            popup=f"MMSI {mmsi}: Predicted (1h)",
        ).add_to(m)

        folium.CircleMarker(
            location=input_coords[0],
            radius=8,
            color=color_input,
            fill=True,
            fillColor="green",
            fillOpacity=1.0,
            popup=f"Start: MMSI {mmsi}",
        ).add_to(m)

        folium.CircleMarker(
            location=actual_coords[-1],
            radius=8,
            color="blue",
            fill=True,
            fillColor="blue",
            fillOpacity=1.0,
            popup=f"Actual End: MMSI {mmsi}",
        ).add_to(m)

        folium.CircleMarker(
            location=pred_coords[-1],
            radius=8,
            color="red",
            fill=True,
            fillColor="red",
            fillOpacity=1.0,
            popup=f"Predicted End: MMSI {mmsi}",
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 160px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p><strong>Legend (Encoder-Decoder)</strong></p>
    <p><span style="color: green;">●</span> Input trajectory (2h)</p>
    <p><span style="color: blue;">━</span> Actual trajectory (1h)</p>
    <p><span style="color: red;">╍</span> Predicted trajectory (1h)</p>
    <p><span style="color: green;">●</span> Start point</p>
    <p><span style="color: blue;">●</span> Actual end | <span style="color: red;">●</span> Predicted end</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    plugins.Fullscreen().add_to(m)

    m.save(OUTPUT_HTML)
    print(f"Saved interactive map to {OUTPUT_HTML}")


def main():
    print("=" * 70)
    print("TRAJECTORY PREDICTION VISUALIZATION (ENCODER-DECODER)")
    print("=" * 70)

    model, config, input_scaler, output_scaler = load_model_and_config(MODEL_PATH)

    df = load_trajectory_data(DATA_DIR, PARQUET_FILES, n_vessels=N_VESSELS)

    sequences, targets, mmsi_list, full_trajectories, timestamps_list = create_3hour_sequences(
        df, config, n_vessels=N_VESSELS
    )

    if len(sequences) == 0:
        print("\nNo valid sequences found. Try different parquet files or reduce min length requirements.")
        return

    predictions = predict_trajectories(model, sequences, input_scaler, output_scaler)

    print("\nCalculating prediction errors...")
    output_timesteps = predictions.shape[1] // 2
    for i, mmsi in enumerate(mmsi_list):
        actual = targets[i].reshape(output_timesteps, 2)
        pred = predictions[i].reshape(output_timesteps, 2)
        error = np.linalg.norm(actual - pred, axis=1)
        print(f"  MMSI {mmsi}: Mean error = {error.mean():.4f}°, Max error = {error.max():.4f}°")

    plot_trajectories(full_trajectories, predictions, mmsi_list, config["output_hours"], timestamps_list)

    create_interactive_map(full_trajectories, predictions, mmsi_list, config["output_hours"])

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print(f"  Matplotlib plot: trajectory_predictions_encoder_decoder.png")
    print(f"  Interactive map: {OUTPUT_HTML}")
    print("=" * 70)


if __name__ == "__main__":
    main()

