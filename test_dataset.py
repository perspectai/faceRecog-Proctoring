import os
import shutil
from pathlib import Path
import random

def split_into_test_dataset(training_dir, test_dir):
    """
    Splits the dataset by moving 3 randomly selected images from each subfolder
    of the training dataset (with 10 or more images) to a test dataset, preserving
    the directory structure.
    
    Args:
    - training_dir (str): Path to the training dataset directory.
    - test_dir (str): Path to the directory where the test dataset will be created.
    """
    # Ensure the test directory exists
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    
    # Iterate through each subfolder in the training directory
    for subfolder in os.listdir(training_dir):
        subfolder_path = os.path.join(training_dir, subfolder)
        
        # Check if it's actually a directory
        if os.path.isdir(subfolder_path):
            images = os.listdir(subfolder_path)
            # Proceed if the subfolder contains 10 or more images
            if len(images) >= 18:
                # Randomly select 3 images
                selected_images = random.sample(images, 3)
                
                # Ensure the corresponding subfolder exists in the test directory
                test_subfolder_path = os.path.join(test_dir, subfolder)
                Path(test_subfolder_path).mkdir(parents=True, exist_ok=True)
                
                # Move the selected images to the test directory
                for image in selected_images:
                    source_path = os.path.join(subfolder_path, image)
                    destination_path = os.path.join(test_subfolder_path, image)
                    shutil.move(source_path, destination_path)
                    
    print("Dataset splitting completed.")

# Specify your training and test dataset directories
training_directory_path = 'train_ds'
test_directory_path = 'test_ds'

# Split the dataset into training and test
split_into_test_dataset(training_directory_path, test_directory_path)


