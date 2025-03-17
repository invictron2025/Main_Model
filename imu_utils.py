import numpy as np

# Earth's radius in meters (approximate)
EARTH_RADIUS = 6378137  

def estimate_position(prev_lat, prev_lon, velocity, acceleration, heading, dt):
    """
    Estimate the new latitude and longitude based on velocity, acceleration, and heading.

    Parameters:
    - prev_lat (float): Previous latitude in degrees.
    - prev_lon (float): Previous longitude in degrees.
    - velocity (float): Current velocity (m/s).
    - acceleration (float): Current acceleration (m/s²).
    - heading (float): Heading in degrees (0 = North, 90 = East, 180 = South, 270 = West).
    - dt (float): Time step in seconds.

    Returns:
    - (new_lat, new_lon): Updated latitude and longitude.
    """
    # Convert heading to radians
    heading_rad = np.radians(heading)

    # Compute displacement using kinematics: d = v * t + 0.5 * a * t²
    displacement = (velocity * dt) + (0.5 * acceleration * dt**2)

    # Compute delta in meters
    delta_x = displacement * np.cos(heading_rad)  # East-West movement
    delta_y = displacement * np.sin(heading_rad)  # North-South movement

    # Convert meter displacement to latitude and longitude change
    delta_lat = (delta_y / EARTH_RADIUS) * (180 / np.pi)
    delta_lon = (delta_x / (EARTH_RADIUS * np.cos(np.radians(prev_lat)))) * (180 / np.pi)

    # Compute new coordinates
    new_lat = prev_lat + delta_lat
    new_lon = prev_lon + delta_lon

    return new_lat, new_lon
