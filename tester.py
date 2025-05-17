import os
import time
import shutil
import cosysairsim as airsim
import match_query_imu
import csv


client = airsim.MultirotorClient()
client.confirmConnection()
state = client.getMultirotorState()
# position = state.kinematics_estimated.position

# Define directories
test_directory = "./Data/hall10_data/query_drone/0"  # Replace with your test directory path
processed_directory = "./Data/hall10_data/processed"  # Replace with your processed directory path
script_path = "./match_query_imu.py"  # Replace with actual path to match_query_imu.py
result_file_path = "./result.csv"
result_file = open(result_file_path,mode='w')
writer = csv.writer(result_file)
last_time = time.time()
count = 0

def get_location():
    gps = client.getGpsData()
    return (gps.gnss.geo_point.latitude, gps.gnss.geo_point.longitude)

last_pos = get_location()
# Make sure processed directory exists
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

def save_image():
    global count
    print("Trying to save image")
    responses = client.simGetImages([
    airsim.ImageRequest("downward_cam", airsim.ImageType.Scene,False,True)])
    for r in responses:
        airsim.write_file(os.path.normpath(f"{test_directory}/{count}.png"),r.image_data_uint8)
    print("Saved Image")
    count+=1

def run_test_model(image_path):
    try:
        global last_time,last_pos
        dt = time.time()-last_time
        state = client.getMultirotorState()
        kinematics = state.kinematics_estimated
        orientation = kinematics.orientation
# Convert quaternion to yaw angle in radians
        yaw = airsim.quaternion_to_euler_angles(orientation)[2]  # returns (roll, pitch, yaw)
        gps_pos = get_location()
        last_pos = match_query_imu.main(last_known_position=last_pos,velocity=kinematics.linear_velocity.get_length(),acceleration=kinematics.linear_acceleration.get_length(),heading=yaw,dt=dt)
        writer.writerow([gps_pos[0],gps_pos[1],last_pos[0],last_pos[1]])
        last_time = time.time()
        return True
    except Exception as e:
        print(f"Error running match_query_imu.py: {str(e)}")
        return False

def is_image_file(filename):
    # Check if file is an image based on common extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return filename.lower().endswith(image_extensions)

def monitor_directory():
    while True:
        save_image()
        # Get list of files in test directory
        files = os.listdir(test_directory)
        
        # Filter for image files
        image_files = [f for f in files if is_image_file(f)]
        
        if image_files:
            # Process each image found
            for image_file in image_files:
                source_path = os.path.join(test_directory, image_file)
                destination_path = os.path.join(processed_directory, image_file)
                try:

                    # Run match_query_imu.py on the image
                    start = time.time()
                    run_status = run_test_model(source_path)
                    end = time.time()
                    print("Elapsed Time in seconds",end-start)
                    if run_status:
                        # Move the image to processed directory
                        shutil.move(source_path, destination_path)
                        print(f"Moved {image_file} to processed directory")
                        print("Monitoring the directory...")
                    else:
                        print(f"Failed to process {image_file}")
                except Exception as e:
                    print(f"Error processing {image_file}: {str(e)}")
        
        # Wait for 0.1 second before checking again
        time.sleep(0.1)

def main():
    print("Starting at pos",last_pos)
    print("Monitoring the directory...")
    try:
        monitor_directory()
    except KeyboardInterrupt:
        result_file.close()
        print("\nMonitoring stopped by user")


if __name__ == "__main__":
    main()