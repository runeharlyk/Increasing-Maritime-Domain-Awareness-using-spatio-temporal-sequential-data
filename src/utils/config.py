# Geographic bounding box filter [north, west, south, east]
BOUNDING_BOX = [60, 0, 50, 20]

# Vessel class filter
VESSEL_CLASSES = ["Class A", "Class B"]

# Ship type filter
SHIP_TYPES = ["Cargo", "Tanker", "Passenger"]

# MMSI validation
MMSI_LENGTH = 9
MMSI_MID_MIN = 200  # Minimum Maritime Identification Digits
MMSI_MID_MAX = 775  # Maximum Maritime Identification Digits

# Track filtering thresholds
TRACK_MIN_LENGTH = 100  # Minimum number of datapoints per track/segment
TRACK_MIN_SOG = 5  # Minimum SOG in knots
TRACK_MAX_SOG = 60  # Maximum SOG in knots
TRACK_MIN_TIMESPAN = 60 * 60  # Minimum timespan in seconds (1 hour)

# Segment creation
SEGMENT_TIME_GAP = 15 * 60  # Maximum time gap before creating new segment (15 minutes in seconds)

# Unit conversions
KNOTS_TO_MS = 0.514444  # Conversion factor from knots to m/s

# Point-to-point speed validation
MIN_POINT_TO_POINT_SPEED_KMH = 5
MAX_POINT_TO_POINT_SPEED_KMH = 110
SPEED_ANOMALY_ACTION = "drop"

MIN_DISTANCE_KM = 5  # Minimum km per hour that must be traveled in each sequence

MIN_DISTANCE_KM_OUTPUT = 3

CHECK_OUTPUT_DISTANCE_SEPARATELY = True

# Feature Engineering
FEATURE_COLS = [
    "Latitude",
    "Longitude",
    "SOG",
    "COG_sin",
    "COG_cos",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "SOG_diff",
    "COG_diff_sin",
    "COG_diff_cos",
]

# Columns to apply StandardScaler to (others are assumed to be sin/cos or categorical)
COLS_TO_NORMALIZE = ["Latitude", "Longitude", "SOG", "SOG_diff"]

# Output settings
OUTPUT_FILE = "filtered_vessels.parquet"
COMPRESSION = "snappy"
