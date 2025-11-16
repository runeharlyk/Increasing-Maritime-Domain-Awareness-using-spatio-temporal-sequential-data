import random
import numpy as np
import folium
from folium import plugins
import matplotlib.pyplot as plt


def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))


def create_trajectory_map(df, output_path, max_vessels=100, center_lat=None, center_lon=None, zoom_start=7):
    import polars as pl
    
    if center_lat is None:
        center_lat = df["Latitude"].mean()
    if center_lon is None:
        center_lon = df["Longitude"].mean()

    print(f"\nCreating map centered at ({center_lat:.4f}, {center_lon:.4f})...")
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")

    unique_vessels = df["MMSI"].unique().to_list()
    if len(unique_vessels) > max_vessels:
        print(f"Limiting display to {max_vessels} vessels out of {len(unique_vessels)}")
        selected_vessels = random.sample(unique_vessels, max_vessels)
    else:
        selected_vessels = unique_vessels
        print(f"Displaying all {len(selected_vessels)} vessels")

    df_filtered = df.filter(pl.col("MMSI").is_in(selected_vessels))

    # Group by vessel and segment
    group_cols = ["MMSI", "Segment"]
    if "FileIndex" in df_filtered.columns:
        group_cols.insert(1, "FileIndex")
    
    segments_df = df_filtered.group_by(group_cols).agg(
        [
            pl.col("Latitude").alias("lats"),
            pl.col("Longitude").alias("lons"),
            pl.col("Timestamp").min().alias("start_time"),
            pl.col("Timestamp").max().alias("end_time"),
            pl.len().alias("num_points"),
        ]
    )

    print(f"\nAdding {len(segments_df)} trajectory segments to map...")

    for row in segments_df.iter_rows(named=True):
        mmsi = row["MMSI"]
        segment = row["Segment"]
        lats = row["lats"]
        lons = row["lons"]
        num_points = row["num_points"]

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
            coordinates, color=color, weight=2, opacity=0.7, popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

        # Add start marker
        folium.CircleMarker(
            location=coordinates[0],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            popup=f"Start: {mmsi}",
        ).add_to(m)

        # Add end marker
        folium.CircleMarker(
            location=coordinates[-1],
            radius=3,
            color=color,
            fill=True,
            fillColor="red",
            fillOpacity=0.9,
            popup=f"End: {mmsi}",
        ).add_to(m)

    plugins.Fullscreen().add_to(m)

    m.save(output_path)
    print(f"\nMap saved to {output_path}")
    
    return m


def plot_trajectory_comparison(full_trajectories, predictions, mmsi_list, output_hours, output_path=None):
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

        # Plot trajectories
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
        axes[i, 0].set_title(f"Vessel {mmsi_list[i]} - Trajectory Prediction", fontsize=12)
        axes[i, 0].legend(fontsize=9, loc="best")
        axes[i, 0].grid(True, alpha=0.3)

        # Plot error
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
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved matplotlib plot to {output_path}")

    return fig, axes


def create_prediction_map(full_trajectories, predictions, mmsi_list, output_hours, output_path=None):
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

        color_input = generate_random_color()

        # Input trajectory
        input_coords = [[lat, lon] for lat, lon in input_traj]
        folium.PolyLine(
            input_coords,
            color=color_input,
            weight=4,
            opacity=0.8,
            popup=f"MMSI {mmsi}: Input (2h)",
        ).add_to(m)

        # Actual trajectory
        actual_coords = [[lat, lon] for lat, lon in output_traj]
        folium.PolyLine(
            actual_coords,
            color="blue",
            weight=3,
            opacity=0.7,
            popup=f"MMSI {mmsi}: Actual (1h)",
        ).add_to(m)

        # Predicted trajectory
        pred_coords = [[lat, lon] for lat, lon in pred_traj]
        folium.PolyLine(
            pred_coords,
            color="red",
            weight=3,
            opacity=0.7,
            dash_array="10, 5",
            popup=f"MMSI {mmsi}: Predicted (1h)",
        ).add_to(m)

        # Markers
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

    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 160px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p><strong>Legend</strong></p>
    <p><span style="color: green;">●</span> Input trajectory (2h)</p>
    <p><span style="color: blue;">━</span> Actual trajectory (1h)</p>
    <p><span style="color: red;">╍</span> Predicted trajectory (1h)</p>
    <p><span style="color: green;">●</span> Start point</p>
    <p><span style="color: blue;">●</span> Actual end | <span style="color: red;">●</span> Predicted end</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    plugins.Fullscreen().add_to(m)

    if output_path is not None:
        m.save(output_path)
        print(f"Saved interactive map to {output_path}")
    
    return m

