import polars as pl
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def load_and_prepare_data(data_dir):
    print("Loading data...")

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for file in tqdm(parquet_files, desc="Loading files"):
        try:
            df_temp = pl.read_parquet(file)
            dfs.append(df_temp)
        except Exception as e:
            print(f"Warning: Failed to load {file.name}: {e}")

    if not dfs:
        raise ValueError("No data loaded successfully")

    df = pl.concat(dfs, how="vertical")
    print(f"Loaded {len(df)} rows from {len(dfs)} files")

    required_cols = ["MMSI", "Latitude", "Longitude", "SOG", "COG"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Timestamp" not in df.columns:
        print("Warning: No Timestamp column found. Creating synthetic timestamps...")
        print("Note: For real training, you should include timestamps in clean_polars.py")
        df = df.sort("MMSI")

        timestamps = []
        for mmsi in df["MMSI"].unique().to_list():
            vessel_df = df.filter(pl.col("MMSI") == mmsi)
            n_points = len(vessel_df)
            timestamps.extend([pl.datetime(2025, 1, 1) + pl.duration(minutes=i) for i in range(n_points)])

        df = df.with_columns(pl.Series("Timestamp", timestamps))
    else:
        if df["Timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("Timestamp").cast(pl.Datetime))

    df = df.sort(["MMSI", "Timestamp"])

    return df


def create_sequences(df, input_hours, output_hours, sampling_rate):
    print(f"\nCreating sequences ({input_hours}h input -> {output_hours}h output)...")

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    feature_cols = ["Latitude", "Longitude", "SOG", "COG"]

    print("Resampling with Polars...")

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", by="MMSI")
        .agg(
            [
                pl.col("Latitude").first(),
                pl.col("Longitude").first(),
                pl.col("SOG").first(),
                pl.col("COG").first(),
            ]
        )
        .drop_nulls(subset=feature_cols)
    )

    sequences = []
    targets = []
    mmsi_labels = []

    print("Creating sliding windows...")
    vessel_groups = df_processed.partition_by("MMSI", as_dict=True)

    for mmsi, vessel_data in tqdm(vessel_groups.items(), desc="Processing vessels"):
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(feature_cols).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()

        for i in range(n_points - min_length + 1):
            input_seq = data_array[i : i + input_timesteps]
            output_seq = lat_lon[i + input_timesteps : i + input_timesteps + output_timesteps]

            if not (np.isnan(input_seq).any() or np.isnan(output_seq).any()):
                sequences.append(input_seq)
                targets.append(output_seq.flatten())
                mmsi_labels.append(mmsi[0])

    sequences = np.array(sequences)
    targets = np.array(targets)
    mmsi_labels = np.array(mmsi_labels)

    print(f"Created {len(sequences)} sequences from {len(np.unique(mmsi_labels))} unique vessels")

    return sequences, targets, mmsi_labels, feature_cols


def create_sequences_with_features(df, input_hours, output_hours, sampling_rate):
    print(f"\nCreating sequences ({input_hours}h input -> {output_hours}h output)...")

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    feature_cols = ["Latitude", "Longitude", "SOG", "COG"]

    print("Resampling and adding features with Polars...")

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", by="MMSI")
        .agg(
            [
                pl.col("Latitude").first(),
                pl.col("Longitude").first(),
                pl.col("SOG").first(),
                pl.col("COG").first(),
            ]
        )
        .drop_nulls(subset=feature_cols)
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

    enhanced_features = feature_cols + ["hour", "day_of_week", "SOG_diff", "COG_diff"]

    sequences = []
    targets = []
    mmsi_labels = []

    print("Creating sliding windows...")
    vessel_groups = df_processed.partition_by("MMSI", as_dict=True)

    for mmsi, vessel_data in tqdm(vessel_groups.items(), desc="Processing vessels"):
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(enhanced_features).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()

        for i in range(n_points - min_length + 1):
            input_seq = data_array[i : i + input_timesteps]
            output_seq = lat_lon[i + input_timesteps : i + input_timesteps + output_timesteps]

            if not (np.isnan(input_seq).any() or np.isnan(output_seq).any()):
                sequences.append(input_seq)
                targets.append(output_seq.flatten())
                mmsi_labels.append(mmsi[0])

    sequences = np.array(sequences)
    targets = np.array(targets)
    mmsi_labels = np.array(mmsi_labels)

    print(f"Created {len(sequences)} sequences from {len(np.unique(mmsi_labels))} unique vessels")
    print(f"Input shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")

    return sequences, targets, mmsi_labels, enhanced_features


def split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.8, random_seed=42):
    unique_mmsi = np.unique(mmsi_labels)
    n_vessels = len(unique_mmsi)

    np.random.seed(random_seed)
    shuffled_mmsi = np.random.permutation(unique_mmsi)

    split_idx = int(train_ratio * n_vessels)
    train_mmsi = set(shuffled_mmsi[:split_idx])
    test_mmsi = set(shuffled_mmsi[split_idx:])

    print(f"\nVessel-based split:")
    print(f"  Train vessels: {len(train_mmsi)}")
    print(f"  Test vessels: {len(test_mmsi)}")

    train_mask = np.array([mmsi in train_mmsi for mmsi in mmsi_labels])
    test_mask = np.array([mmsi in test_mmsi for mmsi in mmsi_labels])

    X_train = sequences[train_mask]
    X_test = sequences[test_mask]
    y_train = targets[train_mask]
    y_test = targets[test_mask]

    print(f"  Train sequences: {len(X_train)}")
    print(f"  Test sequences: {len(X_test)}")

    train_vessels_in_data = set(mmsi_labels[train_mask])
    test_vessels_in_data = set(mmsi_labels[test_mask])
    overlap = train_vessels_in_data & test_vessels_in_data

    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} vessels appear in both sets!")
    else:
        print(f"  ✅ No vessel overlap - proper split confirmed!")

    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test, y_train, y_test):
    print("\nNormalizing data...")

    input_scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = input_scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)

    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = input_scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)

    output_scaler = StandardScaler()
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, input_scaler, output_scaler
