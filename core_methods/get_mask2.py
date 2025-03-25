import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to segment the hand using green background
def segment_hand_with_green_background(image):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define a mask for green background
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create mask where green is present
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Invert mask to segment the hand
    foreground_mask = cv2.bitwise_not(green_mask)

    # Apply the mask to the original image to get the foreground (hand)
    foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

    return foreground, foreground_mask

# Function to segment the hand using contour detection and binarization
def segment_hand_with_binarization_and_contours(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Binarize the image using a threshold
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Use Canny edge detection on the binary image to find edges
    edges = cv2.Canny(binary_image, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw contours
    image_contours = image.copy()

    # Draw the largest contour (assuming it's the hand) on the image
    if contours:
        # Sort contours by area and pick the largest one
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(image_contours, [largest_contour], -1, (255, 0, 0), 2)

    return image_contours, edges, binary_image

# Example usage
# Load the image (replace with your own image path)
image_path = 'assets/hand.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# First method with green background simulation
green_background_image = np.full(image_rgb.shape, [0, 255, 0], dtype=np.uint8)  # Simulated green background
hand_segmented, hand_mask = segment_hand_with_green_background(green_background_image)

# Second method with binarization and contour detection
hand_contour_segmented, hand_edges, binary_image = segment_hand_with_binarization_and_contours(image_rgb)

# Display the results
plt.figure(figsize=(10, 10))

# Display the results for green background segmentation
plt.subplot(2, 3, 1)
plt.imshow(hand_mask, cmap='gray')
plt.title('Mask (Green Background)')

plt.subplot(2, 3, 2)
plt.imshow(hand_segmented)
plt.title('Segmented Hand (Green Background)')

# Display the results for binarization + contour-based segmentation
plt.subplot(2, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Binarized Image')

plt.subplot(2, 3, 4)
plt.imshow(hand_edges, cmap='gray')
plt.title('Edges (Binarized + Contours)')

plt.subplot(2, 3, 5)
plt.imshow(hand_contour_segmented)
plt.title('Segmented Hand (Binarized + Contours)')

plt.show()
