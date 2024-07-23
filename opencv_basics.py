import cv2
# import numpy as np

# Load an image from file
image = cv2.imread('C:/Users/Danya/Downloads/D1web-12.jpg')  # Replace 'path_to_your_image.jpg' with the path to your
# image file

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image")
    exit()

# Display the original image
cv2.imshow('Original Image', image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)

# Perform edge detection using the Canny method
edges = cv2.Canny(gray_image, 100, 200)
cv2.imshow('Edges', edges)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()