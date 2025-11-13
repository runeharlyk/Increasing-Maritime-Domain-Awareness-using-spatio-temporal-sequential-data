import polars as pl
from pathlib import Path
import sys

DATA_DIR = Path("data")
PARQUET_FILE = "aisdk-2024-03-21.parquet"
TARGET_MMSI = "304846000"


def main():
    mmsi = TARGET_MMSI
    if len(sys.argv) > 1:
        mmsi = sys.argv[1]

    parquet_path = DATA_DIR / PARQUET_FILE

    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist")
        return

    print(f"Loading data from {PARQUET_FILE}...")
    df = pl.read_parquet(parquet_path)

    print(f"Filtering for MMSI: {mmsi}...")
    vessel_df = df.filter(pl.col("MMSI") == mmsi)

    if len(vessel_df) == 0:
        print(f"No data found for MMSI {mmsi}")
        print(f"\nSearching for similar MMSI values...")
        all_mmsi = df.select("MMSI").unique().sort("MMSI")
        similar = all_mmsi.filter(pl.col("MMSI").str.contains(mmsi[:6]))
        if len(similar) > 0:
            print(f"Found {len(similar)} similar MMSI values:")
            for m in similar.head(20)["MMSI"]:
                print(f"  - {m}")
        return

    print(f"\nFound {len(vessel_df)} datapoints for MMSI {mmsi}")

    segments = vessel_df.select("Segment").unique().sort("Segment")
    num_segments = len(segments)
    print(f"Number of segments: {num_segments}")

    print(f"\n{'='*80}")
    print(f"SEGMENT DETAILS FOR MMSI {mmsi}")
    print(f"{'='*80}\n")

    for seg_num in segments["Segment"]:
        seg_df = vessel_df.filter(pl.col("Segment") == seg_num)

        print(f"Segment {seg_num}:")
        print(f"  Points: {len(seg_df)}")
        print(f"  Time range: {seg_df['Timestamp'].min()} to {seg_df['Timestamp'].max()}")
        print(f"  Duration: {(seg_df['Timestamp'].max() - seg_df['Timestamp'].min())}")

        if "SOG" in seg_df.columns:
            print(
                f"  SOG (m/s): min={seg_df['SOG'].min():.2f}, max={seg_df['SOG'].max():.2f}, avg={seg_df['SOG'].mean():.2f}"
            )

        if "COG" in seg_df.columns:
            print(f"  COG (degrees): min={seg_df['COG'].min():.1f}, max={seg_df['COG'].max():.1f}")

        lat_min, lat_max = seg_df["Latitude"].min(), seg_df["Latitude"].max()
        lon_min, lon_max = seg_df["Longitude"].min(), seg_df["Longitude"].max()
        print(f"  Latitude range: {lat_min:.4f} to {lat_max:.4f}")
        print(f"  Longitude range: {lon_min:.4f} to {lon_max:.4f}")

        if "Type of mobile" in seg_df.columns:
            print(f"  Type: {seg_df['Type of mobile'][0]}")

        if "Ship type" in seg_df.columns:
            ship_types = seg_df["Ship type"].unique()
            print(f"  Ship type: {', '.join(ship_types)}")

        print()

    output_csv = f"vessel_{mmsi}_segments.csv"
    vessel_df.write_csv(output_csv)
    print(f"\nFull data exported to: {output_csv}")

    summary_df = (
        vessel_df.group_by("Segment")
        .agg(
            [
                pl.len().alias("num_points"),
                pl.col("Timestamp").min().alias("start_time"),
                pl.col("Timestamp").max().alias("end_time"),
                (pl.col("Timestamp").max() - pl.col("Timestamp").min()).dt.total_seconds().alias("duration_seconds"),
                pl.col("SOG").min().alias("min_sog"),
                pl.col("SOG").max().alias("max_sog"),
                pl.col("SOG").mean().alias("avg_sog"),
                pl.col("Latitude").min().alias("min_lat"),
                pl.col("Latitude").max().alias("max_lat"),
                pl.col("Longitude").min().alias("min_lon"),
                pl.col("Longitude").max().alias("max_lon"),
            ]
        )
        .sort("Segment")
    )

    output_summary = f"vessel_{mmsi}_summary.csv"
    summary_df.write_csv(output_summary)
    print(f"Segment summary exported to: {output_summary}")

    print(f"\nSummary table:")
    print(summary_df)


if __name__ == "__main__":
    main()
