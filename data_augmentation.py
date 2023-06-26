import os
import cv2
import numpy as np
import random

# Folder path
folder_path = "C:/Users/ririk/Desktop/20230406_TEMP/DLIP_extra_project/raw_datasets/rock"

# Get the list of image files
file_list = os.listdir(folder_path)

# Number of expansions for each image
num_expansions = 300

# Image expansion
for i in range(num_expansions):
    # Randomly select an image file
    file_name = random.choice(file_list)
    if file_name.endswith(".jpg"):
        # Image file path
        file_path = os.path.join(folder_path, file_name)
        
        # Load the image
        image = cv2.imread(file_path)
        
        # Select the type of transformation to apply
        transformation_type = random.choice(["gaussian", "salt_and_pepper", "hsv"])
        
        # Add Gaussian noise
        if transformation_type == "gaussian":
            mean = 0
            std_dev = 0.03  # Standard deviation of the noise
            noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, noise)
        
        # Add Salt and Pepper noise
        elif transformation_type == "salt_and_pepper":
            noise = np.zeros_like(image)
            salt = np.random.randint(128, 255, (image.shape[0], image.shape[1]), dtype=np.uint8)
            pepper = np.random.randint(128, 255, (image.shape[0], image.shape[1]), dtype=np.uint8)
            noise[salt == 255] = 128  # Add Salt noise
            noise[pepper == 0] = 128  # Add Pepper noise
            noisy_image = cv2.add(image, noise)
        
        # Apply HSV value transformation
        elif transformation_type == "hsv":
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Set random transformation values
            hue_shift = random.randint(-10, 10)
            saturation_scale = random.uniform(0.5, 1.5)
            value_scale = random.uniform(0.5, 1.5)
            
            # Apply transformation to HSV values
            hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255).astype(np.uint8)
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * value_scale, 0, 255).astype(np.uint8)
            
            # Convert the modified image back to BGR format
            noisy_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        # Save the expanded image
        expanded_file_path = os.path.join(folder_path, f"{i + 1001}.jpg")
        cv2.imwrite(expanded_file_path, noisy_image)
