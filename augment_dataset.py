import random

import cv2
import os

# Specify the directory containing your original images
original_dir = r"C:\Users\Stefan\PycharmProjects\pythonProject1\dataset\stug_3"

# Specify the directory where augmented images will be saved
augmented_dir = r"C:\Users\Stefan\PycharmProjects\pythonProject1\augmented\stug_3"

# Create the augmented directory if it doesn't exist
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

for filename in os.listdir(original_dir):
    if filename.endswith(".bmp"):  # Adjust file extensions as needed
        img_path = os.path.join(original_dir, filename)
        img = cv2.imread(img_path)

        # Convert the image to black and white
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the black and white image
        cv2.imwrite(os.path.join(original_dir, f"{filename}"), bw_img)


# Iterate through the original images and perform flipping
for filename in os.listdir(original_dir):
    if filename.endswith(".bmp"):  # Adjust file extensions as needed
        img_path = os.path.join(original_dir, filename)
        img = cv2.imread(img_path)
        flipped_lr = cv2.flip(img, 1)  # Flip horizontally

        # Save the augmented images
        cv2.imwrite(os.path.join(augmented_dir, f"flipped_lr_{filename}"), flipped_lr)

for filename in os.listdir(augmented_dir):
    if filename.endswith(".bmp"):  # Adjust file extensions as needed
        img_path = os.path.join(augmented_dir, filename)
        img = cv2.imread(img_path)
        flipped_tb = cv2.flip(img, 0)  # Flip vertically

        # Save the augmented images
        cv2.imwrite(os.path.join(augmented_dir, f"flipped_tb_{filename}"), flipped_tb)

zoom_factor = 1.2  # Zoom factor, adjust as needed

for filename in os.listdir(augmented_dir):
    if filename.endswith(".bmp"):  # Adjust file extensions as needed
        img_path = os.path.join(augmented_dir, filename)
        img = cv2.imread(img_path)

        # Randomly select a zoom-in point
        height, width, _ = img.shape
        center_x = random.randint(int(width * 0.25), int(width * 0.75))
        center_y = random.randint(int(height * 0.25), int(height * 0.75))

        # Define the transformation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_factor)

        # Apply the transformation
        zoomed_in = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

        # Save the augmented image
        cv2.imwrite(os.path.join(augmented_dir, f"zoomed_in_{filename}"), zoomed_in)