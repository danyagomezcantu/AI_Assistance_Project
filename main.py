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

"""
import cv2
import dlib

# Load the pre-trained face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Download from dlib's website

# Load an image from file
image = cv2.imread('path_to_your_image.jpg')  # Replace with the path to your image file

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector(gray_image)

# Loop over each detected face
for face in faces:
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around the face
    
    # Predict facial landmarks
    landmarks = landmark_predictor(gray_image, face)
    
    # Loop over each landmark point
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Draw a circle for each landmark point

# Display the output
cv2.imshow('Facial Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
