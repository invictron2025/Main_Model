import numpy as np
from pyproj import Geod

# Earth's radius in meters (approximate)
EARTH_RADIUS = 6378137  
geod = Geod(ellps="WGS84")

def estimate_position(prev_lat, prev_lon, velocity, acceleration, heading, dt):
    # Compute displacement using kinematics: d = v * t + 0.5 * a * t²
    displacement = (velocity * dt) + (0.5 * acceleration * dt**2)
    new_lon,new_lat,_ = geod.fwd(prev_lon,prev_lat,np.degrees(heading),displacement)
    return new_lat, new_lon
