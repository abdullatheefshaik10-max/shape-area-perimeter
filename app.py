import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="9-Shape Detector", layout="wide")
st.title("🔷 9-Shape Geometric Analyzer")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def detect_shape_refined(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    vertices = len(approx)
    
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    extent = float(area) / (w * h) if (w * h) > 0 else 0
    
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        if extent > 0.85:
            return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif 0.65 < extent <= 0.85:
            return "Trapezium"
        else:
            return "Diamond"
    elif vertices == 6:
        return "Star" if solidity < 0.8 else "Hexagon"
    elif vertices > 6:
        return "Star" if solidity < 0.8 else "Circle"
    return "Unknown"

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    final_shapes = []

    # Get image dimensions to find the "Main Frame"
    img_height, img_width = img.shape[:2]
    total_img_area = img_height * img_width

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # --- THE FIX ---
        # 1. Ignore small noise/text (Area < 500)
        # 2. Ignore the giant border box (Area > 80% of image)
        if area < 500 or area > (total_img_area * 0.8):
            continue

        shape = detect_shape_refined(cnt)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(result_img, shape, (cx - 40, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        final_shapes.append(shape)

    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.success(f"Successfully Detected: {len(final_shapes)} Shapes")
