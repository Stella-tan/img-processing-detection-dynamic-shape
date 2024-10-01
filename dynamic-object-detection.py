import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import random

# Global variable for minimum area
min_area = 1000

# Function to detect rectangles in the image
def rectangle_detection(image, min_area):
    if image is None:
        st.error("Failed to load image.")
        return

    # Resize the image if it's too large
    window_width, window_height = 800, 600
    aspect_ratio = image.shape[1] / image.shape[0]

    if image.shape[1] > window_width or image.shape[0] > window_height:
        if aspect_ratio > 1:
            new_width = window_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_img = image.copy()

    # Convert to grayscale and threshold the image for contour detection
    frame_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over all contours and find rectangles on the original color image
    rectangle_count = 0
    output_img = resized_img.copy()

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(contour) > min_area:
            cv2.drawContours(output_img, [approx], 0, (0, 255, 0), 2)
            rectangle_count += 1

    # Display the total count of rectangles on the image
    cv2.putText(output_img, f"Total Rectangles: {rectangle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return output_img, rectangle_count

# Function for coin counting and classification
def upload_image_coin_counting(image):
    # Load and resize the image
    if image is None:
        st.error("Failed to load image.")
        return None, 0, 0, 0, 0, 0

    window_width, window_height = 800, 600
    aspect_ratio = image.shape[1] / image.shape[0]

    if image.shape[1] > window_width or image.shape[0] > window_height:
        if aspect_ratio > 1:
            new_width = window_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_img = image.copy()

    # Perform grayscale conversion and coin detection
    input_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for contour detection
    _, binary_image = cv2.threshold(input_img_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert grayscale image to color for visualization
    input_img_color = cv2.cvtColor(input_img_gray, cv2.COLOR_GRAY2BGR)
    object_count = 0
    total_value = 0
    count_50sen, count_20sen, count_10sen, count_5sen = 0, 0, 0, 0

    # Iterate over each detected contour
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if 500 <= contour_area <= 5000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Classify the coins based on the radius
            if 34 <= radius <= 36:
                coin_value, coin_label = 0.50, "50 sen"
                count_50sen += 1
            elif 31 <= radius <= 33:
                coin_value, coin_label = 0.20, "20 sen"
                count_20sen += 1
            elif 28 <= radius <= 30:
                coin_value, coin_label = 0.10, "10 sen"
                count_10sen += 1
            elif 25 <= radius <= 27:
                coin_value, coin_label = 0.05, "5 sen"
                count_5sen += 1
            else:
                coin_value, coin_label = 0, "Unknown"

            total_value += coin_value
            cv2.circle(input_img_color, center, radius, (0, 255, 0), 2)
            cv2.putText(input_img_color, coin_label, (center[0] - 20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            object_count += 1

    # Return the results
    return input_img_color, object_count, total_value, count_50sen, count_20sen, count_10sen, count_5sen

# Function to count many objects in the image
def count_manyObjects(image, threshold_value, min_area):
    if image is None:
        st.error("Failed to load image. Please check the file.")
        return None, 0

    # Resize the image if it's too large
    window_width, window_height = 800, 600
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > window_width or h > window_height:
        if aspect_ratio > 1:
            new_width = window_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        image_resized = image.copy()

    # Apply FloodFill to remove the background
    foreground = image_resized.copy()
    seed = (10, 10)  # Seed point for flood fill
    mask = np.zeros((foreground.shape[0] + 2, foreground.shape[1] + 2), np.uint8)  # Create mask for flood fill
    cv2.floodFill(foreground, mask, seedPoint=seed, newVal=(0, 0, 0), loDiff=(5, 5, 5), upDiff=(5, 5, 5))

    # Convert the result to grayscale for processing
    gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)

    # Apply threshold to the grayscale image using the given threshold value
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply morphological opening to clean up small noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    # Find contours of objects
    cntrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area to remove small objects
    filtered_contours = [cnt for cnt in cntrs if cv2.contourArea(cnt) > min_area]

    object_count = len(filtered_contours)  # Count the number of filtered contours (objects)

    # Draw all filtered contours on the original image
    output_image = image_resized.copy()
    for cnt in filtered_contours:
        cv2.drawContours(output_image, [cnt], 0, (0, 255, 0), 2)

    return output_image, object_count  # Return the processed image and object count

# Function to process and classify dynamic objects in the image
def process_dynamicImage(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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



















# Streamlit app layout
st.title("Image Analysis Tool")

# Option selection
option = st.selectbox("Choose Detection Option", ("Rectangle Detection", "Coin Counting", "Count Many Objects", "Dynamic Object Classification"))

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tiff"])

# Parameters for count many objects
threshold_value = st.slider("Threshold Value", 0, 255, 127)
min_area_input = st.number_input("Minimum Area (pixels)", min_value=1, value=100)

if uploaded_file is not None:
    # Read the image using OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Show the original image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    if option == "Rectangle Detection":
        # Detect rectangles
        output_img, rectangle_count = rectangle_detection(image, min_area_input)

        # Show the image with rectangles detected
        st.image(output_img, channels="BGR", caption=f"Detected Rectangles: {rectangle_count}", use_column_width=True)

    elif option == "Coin Counting":
        # Count coins
        output_img_color, object_count, total_value, count_50sen, count_20sen, count_10sen, count_5sen = upload_image_coin_counting(image)

        # Show the image with coins counted
        st.image(output_img_color, channels="BGR", caption="Coin Counting Results", use_column_width=True)

        # Display results
        st.write(f"Total number of coins found: {object_count}")
        st.write(f"Total value of the coins: RM {total_value:.2f}")
        st.write(f"50 sen coins: {count_50sen}, 20 sen coins: {count_20sen}, 10 sen coins: {count_10sen}, 5 sen coins: {count_5sen}")

    elif option == "Count Many Objects":
        # Count many objects
        output_img, object_count = count_manyObjects(image, threshold_value, min_area_input)

        # Show the image with objects counted
        st.image(output_img, channels="BGR", caption=f"Total Objects: {object_count}", use_column_width=True)

    elif option == "Dynamic Object Classification":
        # Process and classify dynamic objects
        process_dynamicImage(image)
