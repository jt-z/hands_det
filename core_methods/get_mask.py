import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'assets/hand.png'
image = cv2.imread(image_path)

# Convert to RGB (OpenCV loads images as BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define a function to segment the hand based on color range (assuming the background can be pure white)
def segment_foreground(image):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define a mask for white background
    lower_white = np.array([0, 0, 200])  # Lower bound for white
    upper_white = np.array([180, 20, 255])  # Upper bound for white

    # Create mask where white is present
    cur_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Invert mask to segment the hand
    foreground_mask = cv2.bitwise_not(cur_mask)

    # Apply the mask to the original image to get the foreground (hand)
    foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

    return foreground, foreground_mask

# Simulate a pure white background and segment the hand
current_background_image = np.full(image_rgb.shape, [255, 255, 255], dtype=np.uint8)  # Simulated white background
hand_segmented, hand_mask = segment_foreground(current_background_image)

# Display the mask and the segmented hand
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(hand_mask, cmap='gray')
plt.title('Foreground Mask (Hand)')
plt.subplot(1, 2, 2)
plt.imshow(hand_segmented)
plt.title('Segmented Hand')
plt.show()
