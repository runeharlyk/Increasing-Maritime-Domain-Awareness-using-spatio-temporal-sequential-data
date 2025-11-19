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
    for idx, file in enumerate(tqdm(parquet_files, desc="Loading files")):
        try:
            df_temp = pl.read_parquet(file)
            df_temp = df_temp.with_columns(pl.lit(idx).alias("FileIndex"))
            dfs.append(df_temp)
        except Exception as e:
            print(f"Warning: Failed to load {file.name}: {e}")

    if not dfs:
        raise ValueError("No data loaded successfully")

    df = pl.concat(dfs, how="vertical")
    print(f"Loaded {len(df)} rows from {len(dfs)} files")

    required_cols = ["MMSI", "Latitude", "Longitude", "SOG", "COG", "Segment", "Timestamp"]
    missing = [col for col in required_cols if col not in df.columns]
    assert len(missing) == 0, f"Missing required columns: {missing}"
    
    assert df["Segment"].is_not_null().all(), "Segment contains null values"
    assert df["Timestamp"].is_not_null().all(), "Timestamp contains null values"

    df = df.sort(["MMSI", "Timestamp"])

    return df


def create_sequences(df, input_hours, output_hours, sampling_rate):
    print(f"\nCreating sequences ({input_hours}h input -> {output_hours}h output)...")

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    feature_cols = ["Latitude", "Longitude", "SOG", "COG"]

    print("Resampling with Polars...")
    
    agg_cols = [
        pl.col("Latitude").first(),
        pl.col("Longitude").first(),
        pl.col("SOG").first(),
        pl.col("COG").first(),
        pl.col("Segment").first(),
    ]
    
    if "FileIndex" in df.columns:
        agg_cols.append(pl.col("FileIndex").first())

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", by="MMSI")
        .agg(agg_cols)
        .drop_nulls(subset=feature_cols)
        .with_columns(
            [
                (pl.col("COG") * np.pi / 180.0).sin().alias("COG_sin"),
                (pl.col("COG") * np.pi / 180.0).cos().alias("COG_cos"),
            ]
        )
    )
    
    feature_cols = ["Latitude", "Longitude", "SOG", "COG_sin", "COG_cos"]

    sequences = []
    targets = []
    mmsi_labels = []

    stride = output_timesteps
    print(f"Creating sequences with stride={stride}")
    
    group_cols = ["MMSI", "Segment"]
    if "FileIndex" in df_processed.columns:
        group_cols.insert(1, "FileIndex")
        print(f"Processing by {group_cols} to avoid crossing trajectory gaps and date boundaries...")
    else:
        print(f"Processing by {group_cols} to avoid crossing trajectory gaps...")
    
    segment_groups = df_processed.partition_by(group_cols, as_dict=True)

    for group_key, vessel_data in tqdm(segment_groups.items(), desc="Processing segments"):
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(feature_cols).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()
        
        mmsi = group_key[0] if isinstance(group_key, tuple) else group_key

        for i in range(0, n_points - min_length + 1, stride):
            input_seq = data_array[i : i + input_timesteps]
            output_seq = lat_lon[i + input_timesteps : i + input_timesteps + output_timesteps]

            if not (np.isnan(input_seq).any() or np.isnan(output_seq).any()):
                sequences.append(input_seq)
                targets.append(output_seq.flatten())
                mmsi_labels.append(mmsi)

    sequences = np.array(sequences)
    targets = np.array(targets)
    mmsi_labels = np.array(mmsi_labels)

    print(f"Created {len(sequences)} sequences from {len(np.unique(mmsi_labels))} unique vessels")
    print(f"  Stride: {stride} timesteps ({stride * sampling_rate} minutes)")
    overlap_pct = max(0, (input_timesteps - stride) / input_timesteps * 100)
    print(f"  Sequence overlap: ~{overlap_pct:.1f}%")

    return sequences, targets, mmsi_labels, feature_cols


def create_sequences_with_features(df, input_hours, output_hours, sampling_rate):
    print(f"\nCreating sequences ({input_hours}h input -> {output_hours}h output)...")

    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    min_length = input_timesteps + output_timesteps

    feature_cols = ["Latitude", "Longitude", "SOG", "COG"]

    print("Resampling and adding features with Polars...")
    
    agg_cols = [
        pl.col("Latitude").first(),
        pl.col("Longitude").first(),
        pl.col("SOG").first(),
        pl.col("COG").first(),
        pl.col("Segment").first(),
    ]
    
    if "FileIndex" in df.columns:
        agg_cols.append(pl.col("FileIndex").first())

    df_processed = (
        df.sort(["MMSI", "Timestamp"])
        .group_by_dynamic("Timestamp", every=f"{sampling_rate}m", by="MMSI")
        .agg(agg_cols)
        .drop_nulls(subset=feature_cols)
        .with_columns(
            [
                (pl.col("Timestamp").dt.hour() / 24.0).alias("hour"),
                (pl.col("Timestamp").dt.weekday() / 7.0).alias("day_of_week"),
                pl.col("SOG").diff().over("MMSI").fill_null(0).alias("SOG_diff"),
            ]
        )
        .with_columns(
            [
                (pl.col("COG") * np.pi / 180.0).sin().alias("COG_sin"),
                (pl.col("COG") * np.pi / 180.0).cos().alias("COG_cos"),
                (((pl.col("COG").diff().over("MMSI").fill_null(0) + 180) % 360) - 180).alias("COG_diff_raw"),
            ]
        )
        .with_columns(
            [
                (pl.col("COG_diff_raw") * np.pi / 180.0).sin().alias("COG_diff_sin"),
                (pl.col("COG_diff_raw") * np.pi / 180.0).cos().alias("COG_diff_cos"),
            ]
        )
    )

    enhanced_features = ["Latitude", "Longitude", "SOG", "COG_sin", "COG_cos", 
                        "hour", "day_of_week", "SOG_diff", "COG_diff_sin", "COG_diff_cos"]

    sequences = []
    targets = []
    mmsi_labels = []

    stride = output_timesteps
    print(f"Creating sequences with stride={stride}")
    
    group_cols = ["MMSI", "Segment"]
    if "FileIndex" in df_processed.columns:
        group_cols.insert(1, "FileIndex")
        print(f"Processing by {group_cols} to avoid crossing trajectory gaps and date boundaries...")
    else:
        print(f"Processing by {group_cols} to avoid crossing trajectory gaps...")
    
    segment_groups = df_processed.partition_by(group_cols, as_dict=True)

    for group_key, vessel_data in tqdm(segment_groups.items(), desc="Processing segments"):
        n_points = len(vessel_data)

        if n_points < min_length:
            continue

        data_array = vessel_data.select(enhanced_features).to_numpy()
        lat_lon = vessel_data.select(["Latitude", "Longitude"]).to_numpy()
        
        mmsi = group_key[0] if isinstance(group_key, tuple) else group_key

        for i in range(0, n_points - min_length + 1, stride):
            input_seq = data_array[i : i + input_timesteps]
            output_seq = lat_lon[i + input_timesteps : i + input_timesteps + output_timesteps]

            if not (np.isnan(input_seq).any() or np.isnan(output_seq).any()):
                sequences.append(input_seq)
                targets.append(output_seq.flatten())
                mmsi_labels.append(mmsi)

    sequences = np.array(sequences)
    targets = np.array(targets)
    mmsi_labels = np.array(mmsi_labels)

    print(f"Created {len(sequences)} sequences from {len(np.unique(mmsi_labels))} unique vessels")
    print(f"  Input shape: {sequences.shape}")
    print(f"  Target shape: {targets.shape}")
    print(f"  Stride: {stride} timesteps ({stride * sampling_rate} minutes)")
    overlap_pct = max(0, (input_timesteps - stride) / input_timesteps * 100)
    print(f"  Sequence overlap: ~{overlap_pct:.1f}%")

    return sequences, targets, mmsi_labels, enhanced_features


def split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.7, val_ratio=0.15, random_seed=42):
    unique_mmsi = np.unique(mmsi_labels)
    n_vessels = len(unique_mmsi)

    np.random.seed(random_seed)
    shuffled_mmsi = np.random.permutation(unique_mmsi)

    train_end = int(train_ratio * n_vessels)
    val_end = int((train_ratio + val_ratio) * n_vessels)
    
    train_mmsi = set(shuffled_mmsi[:train_end])
    val_mmsi = set(shuffled_mmsi[train_end:val_end])
    test_mmsi = set(shuffled_mmsi[val_end:])

    print(f"\nVessel-based split:")
    print(f"  Train vessels: {len(train_mmsi)} ({train_ratio*100:.0f}%)")
    print(f"  Val vessels: {len(val_mmsi)} ({val_ratio*100:.0f}%)")
    print(f"  Test vessels: {len(test_mmsi)} ({(1-train_ratio-val_ratio)*100:.0f}%)")

    train_mask = np.array([mmsi in train_mmsi for mmsi in mmsi_labels])
    val_mask = np.array([mmsi in val_mmsi for mmsi in mmsi_labels])
    test_mask = np.array([mmsi in test_mmsi for mmsi in mmsi_labels])

    X_train = sequences[train_mask]
    X_val = sequences[val_mask]
    X_test = sequences[test_mask]
    y_train = targets[train_mask]
    y_val = targets[val_mask]
    y_test = targets[test_mask]

    print(f"  Train sequences: {len(X_train)}")
    print(f"  Val sequences: {len(X_val)}")
    print(f"  Test sequences: {len(X_test)}")

    train_vessels_in_data = set(mmsi_labels[train_mask])
    val_vessels_in_data = set(mmsi_labels[val_mask])
    test_vessels_in_data = set(mmsi_labels[test_mask])
    
    overlap_train_val = train_vessels_in_data & val_vessels_in_data
    overlap_train_test = train_vessels_in_data & test_vessels_in_data
    overlap_val_test = val_vessels_in_data & test_vessels_in_data

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"  ⚠️  WARNING: Vessel overlap detected!")
        if overlap_train_val:
            print(f"    Train-Val overlap: {len(overlap_train_val)} vessels")
        if overlap_train_test:
            print(f"    Train-Test overlap: {len(overlap_train_test)} vessels")
        if overlap_val_test:
            print(f"    Val-Test overlap: {len(overlap_val_test)} vessels")
    else:
        print(f"  ✅ No vessel overlap - proper split confirmed!")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_data(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\nNormalizing data...")
    
    input_scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    
    X_train_reshaped = X_train.reshape(-1, n_features)
    input_scaler.fit(X_train_reshaped)
    
    X_train_scaled = input_scaler.transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_val_scaled = input_scaler.transform(X_val_reshaped).reshape(X_val.shape[0], n_timesteps, n_features)
    
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = input_scaler.transform(X_test_reshaped).reshape(X_test.shape[0], n_timesteps, n_features)

    output_scaler = StandardScaler()
    output_scaler.fit(y_train)
    y_train_scaled = output_scaler.transform(y_train)
    y_val_scaled = output_scaler.transform(y_val)
    y_test_scaled = output_scaler.transform(y_test)
    
    print(f"  Input features: {n_features}")
    print(f"  Input scaler - mean range: [{input_scaler.mean_.min():.3f}, {input_scaler.mean_.max():.3f}]")
    print(f"  Input scaler - scale range: [{input_scaler.scale_.min():.3f}, {input_scaler.scale_.max():.3f}]")
    print(f"  Output scaler - mean: {output_scaler.mean_}")
    print(f"  Output scaler - scale: {output_scaler.scale_}")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, input_scaler, output_scaler
