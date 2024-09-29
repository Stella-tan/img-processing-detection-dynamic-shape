import cv2
import numpy as np
import streamlit as st
from PIL import Image
import random

# Function to process uploaded image
def process_image(image):
    try:
        # Resize the image for display if necessary
        max_width = 800
        height, width = image.shape[:2]
        if width > max_width:
            scaling_factor = max_width / width
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            image = cv2.resize(image, new_size)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(thresh, 50, 200)

        # Use morphological transformations to close small gaps in contours
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        size_groups = []
        min_area_threshold = 100  

        # Group contours by size
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                continue

            found_group = False
            for idx, (group_area, count) in enumerate(size_groups):
                mean_area = group_area / count
                if abs(mean_area - area) <= 500:  # Compare with mean area instead of group area
                    size_groups[idx] = (group_area + area, count + 1)
                    found_group = True
                    break

            if not found_group:
                size_groups.append((area, 1))

        # Assign a color to each size group
        colors = [tuple(random.sample(range(256), 3)) for _ in range(len(size_groups))]

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                continue

            # Determine which size group this contour belongs to
            for idx, (group_area, count) in enumerate(size_groups):
                mean_area = group_area / count
                if abs(mean_area - area) <= 800:
                    cv2.drawContours(image, [contour], -1, colors[idx], 2)

                    # Get the center of the contour for labeling
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 0, 0

                    # Add the label on the contour
                    label = chr(65 + idx)
                    cv2.putText(image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                    break

        # Display size group counts and mean areas in the upper-left corner
        y0, dy = 50, 30
        for i, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            size_label = f"Size {chr(65 + i)}: {count} (Mean Area: {mean_area:.1f})"
            cv2.putText(image, size_label, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 3)

        # Convert BGR to RGB for displaying in Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the result using Streamlit's image function
        st.image(image_rgb, caption="Processed Image", use_column_width=True)
    
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# Streamlit app structure
st.title("Image Upload and Object Classification")

# Image upload option
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Option to use the camera
use_camera = st.checkbox("Use Camera")

if uploaded_file is not None:
    # Process and display the uploaded image
    image = np.array(Image.open(uploaded_file).convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    process_image(image)

elif use_camera:
    # Open camera and capture a frame
    cap = cv2.VideoCapture(0)
    if st.button("Capture Image"):
        ret, frame = cap.read()
        if ret:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured Image", use_column_width=True)
            process_image(frame)
        else:
            st.error("Failed to capture image from camera")
    cap.release()
