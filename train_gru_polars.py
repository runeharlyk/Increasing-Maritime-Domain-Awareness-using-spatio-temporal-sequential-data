import polars as pl
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
DATA_PATH = Path("data/filtered_vessels_polars.parquet")
INPUT_HOURS = 2
OUTPUT_HOURS = 1
SAMPLING_RATE = 5  # Sample every N minutes
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
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
            dropout=0.2 if num_layers > 1 else 0
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


def load_and_prepare_data(data_path):
    """Load parquet file and prepare for training."""
    print("Loading data...")
    
    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run clean_polars.py first to generate the filtered data."
        )
    
    df = pl.read_parquet(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Check for required columns
    required_cols = ['MMSI', 'Latitude', 'Longitude', 'SOG', 'COG']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check if timestamp exists, if not create synthetic one
    if 'Timestamp' not in df.columns:
        print("Warning: No Timestamp column found. Creating synthetic timestamps...")
        print("Note: For real training, you should include timestamps in clean_polars.py")
        # Create synthetic timestamps assuming 1 minute intervals
        df = df.sort("MMSI")
        
        # Create synthetic timestamps for each MMSI
        timestamps = []
        for mmsi in df['MMSI'].unique().to_list():
            vessel_df = df.filter(pl.col("MMSI") == mmsi)
            n_points = len(vessel_df)
            timestamps.extend(
                [pl.datetime(2025, 1, 1) + pl.duration(minutes=i) for i in range(n_points)]
            )
        
        df = df.with_columns(pl.Series("Timestamp", timestamps))
    else:
        # Ensure timestamp is datetime
        if df["Timestamp"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("Timestamp").cast(pl.Datetime))
    
    # Sort by MMSI and timestamp
    df = df.sort(["MMSI", "Timestamp"])
    
    return df


def create_sequences(df, input_hours, output_hours, sampling_rate):
    """Create input-output sequences for training with MMSI tracking."""
    print(f"\nCreating sequences ({input_hours}h input -> {output_hours}h output)...")
    
    input_timesteps = int(input_hours * 60 / sampling_rate)
    output_timesteps = int(output_hours * 60 / sampling_rate)
    
    # Features to use for prediction
    feature_cols = ['Latitude', 'Longitude', 'SOG', 'COG']
    
    sequences = []
    targets = []
    mmsi_labels = []  # Track which MMSI each sequence belongs to
    
    unique_mmsi = df['MMSI'].unique().to_list()
    
    for mmsi in tqdm(unique_mmsi, desc="Processing vessels"):
        # Filter for this vessel and convert to pandas for resampling
        vessel_data = df.filter(pl.col('MMSI') == mmsi).to_pandas()
        
        # Resample to consistent time intervals
        vessel_data = vessel_data.set_index('Timestamp')
        vessel_data = vessel_data.resample(f'{sampling_rate}min').first()
        vessel_data = vessel_data.dropna(subset=feature_cols)
        
        if len(vessel_data) < input_timesteps + output_timesteps:
            continue
        
        # Create sliding windows
        for i in range(len(vessel_data) - input_timesteps - output_timesteps + 1):
            input_seq = vessel_data.iloc[i:i + input_timesteps][feature_cols].values
            output_seq = vessel_data.iloc[
                i + input_timesteps:i + input_timesteps + output_timesteps
            ][['Latitude', 'Longitude']].values
            
            # Check for valid sequences (no NaN)
            if not (np.isnan(input_seq).any() or np.isnan(output_seq).any()):
                sequences.append(input_seq)
                targets.append(output_seq.flatten())
                mmsi_labels.append(mmsi)  # Track the vessel
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    mmsi_labels = np.array(mmsi_labels)
    
    print(f"Created {len(sequences)} sequences from {len(np.unique(mmsi_labels))} unique vessels")
    
    return sequences, targets, mmsi_labels, feature_cols


def split_by_vessel(sequences, targets, mmsi_labels, train_ratio=0.8, random_seed=42):
    """Split data ensuring no vessel appears in both train and test sets."""
    
    # Get unique vessels
    unique_mmsi = np.unique(mmsi_labels)
    n_vessels = len(unique_mmsi)
    
    # Shuffle vessels
    np.random.seed(random_seed)
    shuffled_mmsi = np.random.permutation(unique_mmsi)
    
    # Split vessels
    split_idx = int(train_ratio * n_vessels)
    train_mmsi = set(shuffled_mmsi[:split_idx])
    test_mmsi = set(shuffled_mmsi[split_idx:])
    
    print(f"\nVessel-based split:")
    print(f"  Train vessels: {len(train_mmsi)}")
    print(f"  Test vessels: {len(test_mmsi)}")
    
    # Create masks
    train_mask = np.array([mmsi in train_mmsi for mmsi in mmsi_labels])
    test_mask = np.array([mmsi in test_mmsi for mmsi in mmsi_labels])
    
    # Split data
    X_train = sequences[train_mask]
    X_test = sequences[test_mask]
    y_train = targets[train_mask]
    y_test = targets[test_mask]
    
    print(f"  Train sequences: {len(X_train)}")
    print(f"  Test sequences: {len(X_test)}")
    
    # Verify no overlap
    train_vessels_in_data = set(mmsi_labels[train_mask])
    test_vessels_in_data = set(mmsi_labels[test_mask])
    overlap = train_vessels_in_data & test_vessels_in_data
    
    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} vessels appear in both sets!")
    else:
        print(f"  ✅ No vessel overlap - proper split confirmed!")
    
    return X_train, X_test, y_train, y_test


def normalize_data(X_train, X_test, y_train, y_test):
    """Normalize features and targets."""
    print("\nNormalizing data...")
    
    # Normalize input features
    input_scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = input_scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_samples, n_timesteps, n_features)
    
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = input_scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_timesteps, n_features)
    
    # Normalize output (lat/lon pairs)
    output_scaler = StandardScaler()
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, input_scaler, output_scaler


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
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History (Polars)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history_polars.png', dpi=300, bbox_inches='tight')
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
        
        ax.plot(true_traj[:, 1], true_traj[:, 0], 'b-o', label='True', markersize=4)
        ax.plot(pred_traj[:, 1], pred_traj[:, 0], 'r-o', label='Predicted', markersize=4)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Trajectory Prediction {i+1}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('predictions_polars.png', dpi=300, bbox_inches='tight')
    print("Saved predictions to predictions_polars.png")


def main():
    print(f"Using device: {DEVICE}")
    print("Using Polars for data loading\n")
    
    # Load data
    df = load_and_prepare_data(DATA_PATH)
    
    # Create sequences with MMSI tracking
    sequences, targets, mmsi_labels, feature_cols = create_sequences(
        df, INPUT_HOURS, OUTPUT_HOURS, SAMPLING_RATE
    )
    
    # Split by vessel (no vessel appears in both train and test)
    X_train, X_test, y_train, y_test = split_by_vessel(
        sequences, targets, mmsi_labels, train_ratio=0.8
    )
    
    # Normalize
    X_train, X_test, y_train, y_test, input_scaler, output_scaler = normalize_data(
        X_train, X_test, y_train, y_test
    )
    
    # Create datasets and dataloaders
    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    input_size = len(feature_cols)
    output_size = y_train.shape[1]  # Flattened lat/lon sequence
    
    model = GRUTrajectoryPredictor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size
    ).to(DEVICE)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    print("Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate_model(model, test_loader, criterion, DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'input_scaler': input_scaler,
                'output_scaler': output_scaler,
                'config': {
                    'input_size': input_size,
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'output_size': output_size,
                    'input_hours': INPUT_HOURS,
                    'output_hours': OUTPUT_HOURS,
                    'sampling_rate': SAMPLING_RATE,
                }
            }, 'best_model_polars.pt')
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, test_loader, output_scaler, DEVICE)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Model saved to: best_model_polars.pt")
    print(f"Training history saved to: training_history_polars.png")
    print(f"Predictions saved to: predictions_polars.png")
    print("="*60)


if __name__ == "__main__":
    main()

