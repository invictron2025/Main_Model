import pandas as pd
from pyproj import Geod
import matplotlib.pyplot as plt

# Load the data (no headers)
df = pd.read_csv("result.csv", header=None)
df.columns = ['lat_orig', 'lon_orig', 'lat_pred', 'lon_pred']

# Initialize geodetic calculator
geod = Geod(ellps="WGS84")

# Compute distances (errors between original and predicted)
def compute_error(row):
    _, _, dist = geod.inv(
        row['lon_orig'], row['lat_orig'],
        row['lon_pred'], row['lat_pred']
    )
    return dist

# Compute distance (error)
df['error_m'] = df.apply(compute_error, axis=1)

# Compute movement distance (e.g. cumulative from first point)
# Here, we assume "distance moved" is distance from the first original point
lat0, lon0 = df['lat_orig'].iloc[0], df['lon_orig'].iloc[0]

def compute_movement(row):
    _, _, dist = geod.inv(
        lon0, lat0,
        row['lon_orig'], row['lat_orig']
    )
    return dist

df['moved_m'] = df.apply(compute_movement, axis=1)

# ✅ Plot
plt.figure(figsize=(10, 6))
plt.plot(df['moved_m'], df['error_m'], label='Error vs Distance', color='blue')
plt.xlabel('Distance Moved (m)')
plt.ylabel('Prediction Error (m)')
plt.title('Prediction Error vs Distance Moved')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
