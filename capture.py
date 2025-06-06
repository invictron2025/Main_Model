import requests
import os
import numpy as np
import csv

# 🔑 Google Maps API Key (Replace with your own valid key)
API_KEY = "API_KEY_HERE"  # Replace with your actual Google Maps API key

# 🌍 Google Maps Settings
ZOOM = 22  # High resolution
TILE_SIZE = 640  # Image tile size (pixels)
OVERLAP_PERCENT = 0.80  # 80% overlap
MOVE_PERCENT = 1 - OVERLAP_PERCENT
EARTH_RADIUS = 6378137  # meters
LATITUDE = 26.50  # Approx latitude for IIT Kanpur
NUM_PATHS = 5  # Total paths (2 left, 1 center, 2 right)

# 📂 Save Directory
SAVE_DIR = "./Data/gallery_satellite/0"
os.makedirs(SAVE_DIR, exist_ok=True)

# 📄 Output CSV Path
CSV_PATH = './Data/gallery_satellite_image_coordinates_grid.csv'

# 📌 Load waypoints from CSV
def load_waypoints(csv_file):
    waypoints = []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            lat, lon = map(float, row)
            waypoints.append((lat, lon))
    return waypoints

# 🧭 Ground resolution at zoom level
def compute_ground_resolution(lat, zoom):
    return (156542.03392 * np.cos(np.radians(lat))) / (2 ** zoom)

# ➖ Lateral shift between adjacent paths
def compute_lateral_shift(zoom):
    resolution = compute_ground_resolution(LATITUDE, zoom)
    image_coverage = resolution * TILE_SIZE
    return image_coverage * MOVE_PERCENT

# 🔁 Interpolate waypoints for image overlap
def interpolate_waypoints(waypoints, zoom):
    resolution = compute_ground_resolution(LATITUDE, zoom)
    image_coverage = resolution * TILE_SIZE
    step_distance = image_coverage * MOVE_PERCENT

    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(a))

    def move_along_line(lat1, lon1, lat2, lon2, step_dist):
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        num_steps = max(1, int(distance / step_dist))
        return [(lat1 + (lat2 - lat1) * i / num_steps, lon1 + (lon2 - lon1) * i / num_steps)
                for i in range(1, num_steps + 1)]

    interpolated = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        interpolated.extend(move_along_line(*waypoints[i], *waypoints[i + 1], step_distance))
    return interpolated

# 🗺️ Get Google Static Map URL
def get_tile_url(lat, lon, zoom, size=TILE_SIZE):
    return f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom={zoom}&size={size}x{size}&maptype=satellite&key={API_KEY}"

# 📍 Generate multiple parallel paths with lateral shift
def generate_parallel_paths(waypoints, zoom):
    lateral_shift = compute_lateral_shift(zoom)
    parallel_paths = []
    for path_offset in range(-(NUM_PATHS // 2), (NUM_PATHS // 2) + 1):
        offset_distance = path_offset * lateral_shift
        shifted = [(lat, lon + (offset_distance / (EARTH_RADIUS * np.cos(np.radians(lat))) * (180 / np.pi)))
                   for lat, lon in waypoints]
        parallel_paths.append(shifted)
    return parallel_paths

# 📥 Download tiles and write metadata
def download_track_images(parallel_paths):
    with open(CSV_PATH, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Image Filename", "Latitude", "Longitude", "Path ID"])

        for path_id, waypoints in enumerate(parallel_paths):
            for i, (lat, lon) in enumerate(waypoints):
                image_filename = f"path{path_id}_waypoint_{i}.png"
                tile_path = os.path.join(SAVE_DIR, image_filename)
                tile_url = get_tile_url(lat, lon, ZOOM)
                response = requests.get(tile_url)

                if response.status_code == 200:
                    with open(tile_path, "wb") as f:
                        f.write(response.content)
                    csv_writer.writerow([image_filename, lat, lon, path_id])
                    print(f"✅ Image saved: {tile_path}")
                else:
                    print(f"❌ Failed to download: {tile_url}")

# 🚀 Entry point
def main():
    csv_file = "./Data/test-path.csv"  # Replace with your own path
    waypoints = load_waypoints(csv_file)
    interpolated_waypoints = interpolate_waypoints(waypoints, ZOOM)
    parallel_paths = generate_parallel_paths(interpolated_waypoints, ZOOM)
    download_track_images(parallel_paths)

if __name__ == '__main__':
    main()
