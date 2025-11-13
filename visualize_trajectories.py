import polars as pl
from pathlib import Path
import folium
from folium import plugins
import random

DATA_DIR = Path("data")
PARQUET_FILE = "aisdk-2024-03-20.parquet"
MAX_VESSELS = 50
OUTPUT_HTML = "trajectories_map.html"

def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def main():
    parquet_path = DATA_DIR / PARQUET_FILE
    
    if not parquet_path.exists():
        print(f"Error: {parquet_path} does not exist")
        print("Available parquet files:")
        for f in sorted(DATA_DIR.glob("*.parquet")):
            print(f"  - {f.name}")
        return
    
    print(f"Loading data from {PARQUET_FILE}...")
    df = pl.read_parquet(parquet_path)
    
    print(f"Total rows: {len(df):,}")
    print(f"Unique vessels (MMSI): {df['MMSI'].n_unique()}")
    print(f"Total segments: {df.select([pl.col('MMSI'), pl.col('Segment')]).unique().height}")
    
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    
    print(f"\nCreating map centered at ({center_lat:.4f}, {center_lon:.4f})...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    unique_vessels = df['MMSI'].unique().to_list()
    if len(unique_vessels) > MAX_VESSELS:
        print(f"Limiting display to {MAX_VESSELS} vessels out of {len(unique_vessels)}")
        selected_vessels = random.sample(unique_vessels, MAX_VESSELS)
    else:
        selected_vessels = unique_vessels
        print(f"Displaying all {len(selected_vessels)} vessels")
    
    df_filtered = df.filter(pl.col('MMSI').is_in(selected_vessels))
    
    segments_df = df_filtered.group_by(['MMSI', 'Segment']).agg([
        pl.col('Latitude').alias('lats'),
        pl.col('Longitude').alias('lons'),
        pl.col('Timestamp').min().alias('start_time'),
        pl.col('Timestamp').max().alias('end_time'),
        pl.len().alias('num_points')
    ])
    
    print(f"\nAdding {len(segments_df)} trajectory segments to map...")
    
    for row in segments_df.iter_rows(named=True):
        mmsi = row['MMSI']
        segment = row['Segment']
        lats = row['lats']
        lons = row['lons']
        num_points = row['num_points']
        
        if len(lats) < 2:
            continue
        
        coordinates = [[lat, lon] for lat, lon in zip(lats, lons)]
        
        color = generate_random_color()
        
        popup_text = f"""
        <b>MMSI:</b> {mmsi}<br>
        <b>Segment:</b> {segment}<br>
        <b>Points:</b> {num_points}<br>
        <b>Start:</b> {row['start_time']}<br>
        <b>End:</b> {row['end_time']}
        """
        
        folium.PolyLine(
            coordinates,
            color=color,
            weight=2,
            opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)
        
        folium.CircleMarker(
            location=coordinates[0],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=f"Start: {mmsi}"
        ).add_to(m)
        
        folium.CircleMarker(
            location=coordinates[-1],
            radius=3,
            color=color,
            fill=True,
            fillColor='red',
            fillOpacity=0.9,
            popup=f"End: {mmsi}"
        ).add_to(m)
    
    plugins.Fullscreen().add_to(m)
    
    m.save(OUTPUT_HTML)
    print(f"\nMap saved to {OUTPUT_HTML}")
    print(f"Open it in your browser to view the trajectories!")

if __name__ == "__main__":
    main()

