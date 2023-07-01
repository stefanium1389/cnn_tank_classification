import random

import cv2
import os

# Specify the directory containing your original images
original_dir = r"C:\Users\Stefan\PycharmProjects\pythonProject1\dataset\tiger"

# Specify the directory where augmented images will be saved
augmented_dir = r"C:\Users\Stefan\PycharmProjects\pythonProject1\augmented\tiger"

# Create the augmented directory if it doesn't exist
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Iterate through the original images and perform flipping
# for filename in os.listdir(original_dir):
#     if filename.endswith(".bmp"):  # Adjust file extensions as needed
#         img_path = os.path.join(original_dir, filename)
#         img = cv2.imread(img_path)
#         flipped_lr = cv2.flip(img, 1)  # Flip horizontally
#
#         # Save the augmented images
#         cv2.imwrite(os.path.join(augmented_dir, f"flipped_lr_{filename}"), flipped_lr)

# for filename in os.listdir(original_dir):
#     if filename.endswith(".bmp"):  # Adjust file extensions as needed
#         img_path = os.path.join(original_dir, filename)
#         img = cv2.imread(img_path)
#         flipped_tb = cv2.flip(img, 0)  # Flip vertically
#
#         # Save the augmented images
#         cv2.imwrite(os.path.join(augmented_dir, f"flipped_tb_{filename}"), flipped_tb)
#
zoom_factor = 1.2  # Zoom factor, adjust as needed
target_width = 250  # Desired width of the zoomed-in region
target_height = 150  # Desired height of the zoomed-in region

for filename in os.listdir(original_dir):
    if filename.endswith(".bmp"):  # Adjust file extensions as needed
        img_path = os.path.join(original_dir, filename)
        img = cv2.imread(img_path)

        # Randomly select a zoom-in point
        height, width, _ = img.shape
        center_x = random.randint(int(width * 0.25), int(width * 0.75))
        center_y = random.randint(int(height * 0.25), int(height * 0.75))

        # Calculate the zoomed-in region boundaries
        zoom_width = int(target_width / zoom_factor)
        zoom_height = int(target_height / zoom_factor)
        zoom_left = max(0, center_x - zoom_width // 2)
        zoom_top = max(0, center_y - zoom_height // 2)
        zoom_right = min(width, zoom_left + zoom_width)
        zoom_bottom = min(height, zoom_top + zoom_height)

        # Crop the zoomed-in region from the original image
        zoomed_in = img[zoom_top:zoom_bottom, zoom_left:zoom_right]

        # Resize the zoomed-in region to the desired size
        zoomed_in = cv2.resize(zoomed_in, (target_width, target_height))

        # Save the augmented image
        cv2.imwrite(os.path.join(augmented_dir, f"zoomed_in_{filename}"), zoomed_in)