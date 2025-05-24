import pandas as pd
import matplotlib.pyplot as plt
import math

def gps_to_meters(lat_ref, lon_ref, lat, lon):
    """Convert GPS coordinates to meters relative to a reference point."""
    R = 6378137.0  # Earth's radius in meters
    dlat = math.radians(lat - lat_ref)
    dlon = math.radians(lon - lon_ref)
    lat_ref_rad = math.radians(lat_ref)

    x = dlon * R * math.cos(lat_ref_rad)
    y = dlat * R
    return x, y

def main(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract columns
    lat_orig = df.iloc[:, 0]
    lon_orig = df.iloc[:, 1]
    lat_pred = df.iloc[:, 2]
    lon_pred = df.iloc[:, 3]

    # Use the first original GPS point as reference
    lat0 = lat_orig.iloc[0]
    lon0 = lon_orig.iloc[0]

    # Convert to meters
    x_orig, y_orig = zip(*[gps_to_meters(lat0, lon0, lat, lon) for lat, lon in zip(lat_orig, lon_orig)])
    x_pred, y_pred = zip(*[gps_to_meters(lat0, lon0, lat, lon) for lat, lon in zip(lat_pred, lon_pred)])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_orig, y_orig, label='Original Path', color='blue', linewidth=2)
    plt.plot(x_pred, y_pred, label='Predicted Path', color='red', linestyle='--', linewidth=2)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Drone Flight Path Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with your CSV file name
    csv_filename = "result.csv"
    main(csv_filename)
