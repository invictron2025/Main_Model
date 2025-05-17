import os
import time
import shutil

import match_query_imu

# Define directories
test_directory = "./Data/hall10_data/query_drone/0"  # Replace with your test directory path
processed_directory = "./Data/hall10_data/processed"  # Replace with your processed directory path
script_path = "./match_query_imu.py"  # Replace with actual path to match_query_imu.py

# Make sure processed directory exists
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

def run_test_model(image_path):
    try:
        match_query_imu.main()
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

if __name__ == "__main__":
    print("Monitoring the directory...")
    try:
        monitor_directory()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")