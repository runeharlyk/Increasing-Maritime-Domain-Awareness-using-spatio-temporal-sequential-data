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
│   ├── models/                # PyTorch Model Definitions (GRU, Attention)
│   ├── utils/                 # Helper functions (Geo-calc, Config, Randomness)
│   └── visualization/         # Plotting and trajectory visualization tools
├── job.sh                     # HPC/LSF submission script
├── models.ipynb               # Jupyter Notebook for experiments and demos
├── requirements.txt           # Python dependencies
├── train.py                   # Main training script
└── trajectory_prediction.py   # Inference and testing script
```

## Getting Started

**Prerequisites**
- Python 3.9+
- PyTorch
- CUDA (optional, for GPU acceleration)

**Installation**
1. Clone the repository:
```bash
git clone https://github.com/your-username/ghost-ship-prediction.git
cd ghost-ship-prediction
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Data Pipeline

The project focuses on Class A and B vessels (Cargo, Tanker, Passenger) within the bounding box 50&deg;-60&deg; N, 0&deg;-20&deg; E.

1. **Download**: Data is sourced from the Danish Maritime Authority (AISDK).

2. **Preprocessing**: Run the cleaning pipeline to filter MMSIs, remove stationary vessels (< 10 knots), and split trajectories into segments



## Models
We implemented and evaluated the following sequence-to-sequence architectures in `src/models/gru_models.py`:

1. **Seq2Seq GRU**: A standard Encoder-Decoder architecture.

2. **Bidirectional GRU with Attention**: An advanced architecture utilizing a custom Attention mechanism to solve the information bottleneck problem, allowing the model to focus on specific historical time steps.

## Training
To train the model locally:

```bash
python train.py
```

**Note: If running on the DTU HPC cluster, utilize the `job.sh` script:**
```bash
bsub < job.sh
```

## Visualization

Use the `models.ipynb` notebook to interactively visualize predictions vs. ground truth. The notebook leverages `src/visualization/trajectories.py` to plot vessel paths on a map.


## Methodology Summary
- Input: Sequence of kinematic data (Lat, Lon, SOG, COG, ROT).
- Regularization: Dropout (0.3), Layer Normalization.
- Training: Teacher Forcing applied to the Decoder.
- Loss Metric: Euclidean Distance (converted from Lat/Lon).

## Authors
- Clara Brodt
- Mads Højgaard
- Rune Harlyk


