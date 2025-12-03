# Maritime Trajectory Prediction using Spatiotemporal Sequential Data

A Deep Learning project for predicting maritime vessel trajectory in the Danish waters (Øresund and Kattegat). This project utilizes Automatic Identification System (AIS) data to forecast future positions of vessels, simulating a "ghost ship" scenario where future identification is masked.

Developed for **DTU Course 02456: Deep Learning**.

## Project Structure

```text
├── data/                      # Data storage
│   ├── aisdk-2024-03-29.csv   # Raw AIS data
│   └── filtered_vessels.csv   # Preprocessed dataset
├── src/                       # Source code
│   ├── data/                  # ETL Pipeline (Download, Clean, Preprocess)
│   ├── models/                # PyTorch Architectures (GRU, GRU+Attention)
│   ├── utils/                 # Helper functions (Geo-calc, Land Masks, Config)
│   └── visualization/         # Plotting and interactive map generation
├── job.sh                     # HPC/LSF submission script
├── trajectory_prediction.ipynb # Main notebook for Training, Evaluation, and Visualization
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```


## Getting Started

**Prerequisites**
- Python 3.9+
- PyTorch
- Polars (for fast data processing)
- CUDA (optional, for GPU acceleration)

**Installation**
1. Clone the repository:
```bash
git clone https://github.com/runeharlyk/Increasing-Maritime-Domain-Awareness-using-spatio-temporal-sequential-data.git
cd Increasing-Maritime-Domain-Awareness-using-spatio-temporal-sequential-data
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data Pipeline

The project focuses on Class A and B vessels (Cargo, Tanker, Passenger) within the bounding box 50&deg;-60&deg; N, 0&deg;-20&deg; E.

1. **Acquisition**: Data is streamed and downloaded directly from the Danish Maritime Authority (AISDK) via the src.data.download module.

2. **Cleaning & Preprocessing**:
- Filters for valid MMSI and vessel types.
- Stationary Filter: Removes vessels moving slower than 5 knots to focus on active trajectories.
- Segmentation: Splits tracks into sequences based on time gaps (15 min).
- Normalization: Standard scaling applied to Latitude, Longitude, and Speed over Ground (SOG).


## Models & Methodology
We implemented sequence-to-sequence (Seq2Seq) architectures to predict a 1-hour future horizon (12 timesteps) based on a 2-hour input history (24 timesteps).

**Architectures**
**Seq2Seq GRU:** A standard Encoder-Decoder architecture using Gated Recurrent Units to capture temporal dependencies.

**GRU with Attention:** An enhanced architecture utilizing a custom Attention mechanism to solve the information bottleneck, allowing the decoder to dynamically weigh specific historical hidden states.

**Key Techniques**
1. **Physics-Informed Loss (Land Mask):** A custom LandMaskHaversineLoss function that combines standard Haversine distance with a heavy penalty for predictions that cross into land masses, ensuring geographically valid trajectories.

2. **Teacher Forcing:** Applied during training with a decay schedule (starts at 1.0, decays to 0.2).

3. **Input Features:** Latitude, Longitude, SOG, Trigonometric COG encodings (Sin/Cos), temporal features, and differentials.

## Results
Both models were trained for 100 epochs. The Attention mechanism demonstrated superior performance across all metrics, particularly in reducing long-term drift (FDE).

| Metric                        | Encoder-Decoder GRU | GRU + Attention | Improvement |
|------------------------------|----------------------|------------------|-------------|
| Mean Haversine Error (MHE)   | 1.29 km              | 1.24 km          | ~3.9%       |
| Root Mean Squared Error (RMSE)| 2.41 km             | 2.33 km          | ~3.3%       |
| Final Displacement Error (FDE)| 2.65 km             | 2.51 km          | ~5.3%       |
| Dynamic Time Warping (DTW)   | 0.62 km              | 0.60 km          | ~3.2%       |


## Usage
**Running Locally**
The primary entry point is the Jupyter Notebook trajectory_prediction.ipynb. It handles:

1. Data downloading and cleaning.

2. Model training loop.

3. Evaluation on the test set.

4. Visualization of predicted paths vs. ground truth.

**HPC Cluster (DTU)**
To submit the training job to the LSF cluster:
```bash
bsub < job.sh
```

**Visualization**
The project generates interactive HTML maps to visualize trajectory predictions.

Blue: Ground Truth

Red: Model Prediction

Green: Input History


## Authors
- Clara Brodt
- Mads Højgaard
- Rune Harlyk


