import os
import time
import shutil
import subprocess

# Define directories
test_directory = "/home/invictron/Model_Work/Main_Model/Data/hall10_data/query_drone/0"  # Replace with your test directory path
processed_directory = "/home/invictron/Model_Work/Main_Model/Data/hall10_data/processed"  # Replace with your processed directory path
script_path = "/home/invictron/Model_Work/Main_Model/match_query_imu.py"  # Replace with actual path to match_query_imu.py

# Make sure processed directory exists
if not os.path.exists(processed_directory):
    os.makedirs(processed_directory)

def run_test_model(image_path):
    try:
        # Run match_query_imu.py with the image path as argument
        result = subprocess.run(['python', script_path, image_path], 
                              capture_output=True, 
                              text=True)
        
        # Print the output from the script
        print(f"Script output: {result.stdout}")
        if result.stderr:
            print(f"Script errors: {result.stderr}")
            
        # Return True if script ran successfully (return code 0)
        return result.returncode == 0
        
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
                    if run_test_model(source_path):
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
    # Verify script exists
    if not os.path.exists(script_path):
        print(f"Error: match_query_imu.py not found at {script_path}")
        exit(1)
        
    print("Monitoring the directory...")
    try:
        monitor_directory()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")