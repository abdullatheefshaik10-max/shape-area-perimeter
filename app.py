import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Refined Shape Detector", layout="wide")
st.title("🔷 Precision Shape & Contour Analyzer")
st.write("Specialized logic for detecting Trapeziums, Diamonds, and Stars.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def detect_shape_refined(cnt):
    # Basic Approximation
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
    vertices = len(approx)
    
    # Area and Bounding Box
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    
    # Extent = Area / Bounding Box Area
    extent = float(area) / (w * h) if (w * h) > 0 else 0
    
    # Solidity = Area / Convex Hull Area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0

    # DETECTION LOGIC
    if vertices == 3:
        return "Triangle"

    elif vertices == 4:
        # High extent (>90%) = Square or Rectangle
        if extent > 0.88:
            return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        # Medium extent (~75%) = Trapezium
        elif 0.65 < extent <= 0.88:
            return "Trapezium"
        # Low extent (~50%) = Diamond
        else:
            return "Diamond"

    elif vertices == 5:
        return "Pentagon"

    elif vertices == 6:
        return "Star" if solidity < 0.7 else "Hexagon"

    elif vertices > 6:
        # Stars have low solidity; Circles have high solidity
        if solidity < 0.7:
            return "Star"
        else:
            return "Circle"

    return "Unknown"

if uploaded_file:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Calculate total image area for filtering the background
    total_img_area = img.shape[0] * img.shape[1]

    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)

    # Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    stats_data = []
    actual_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # --- THE FIX ---
        # 1. Ignore noise (< 400 pixels)
        # 2. Ignore the background container (if it covers > 80% of the image)
        if area < 400 or area > (total_img_area * 0.8):
            continue

        actual_count += 1
        shape = detect_shape_refined(cnt)
        peri = cv2.arcLength(cnt, True)
        
        # Get Center for Text placement
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        # Draw Contours and Labels
        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(result_img, shape, (cx - 40, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        stats_data.append({
            "ID": actual_count,
            "Detected Shape": shape,
            "Area": round(area, 1),
            "Perimeter": round(peri, 1)
        })

    # Layout: Image on left, Table on right
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Visual Result")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Data Analysis")
        st.dataframe(stats_data)
        st.metric("Total Shapes Detected", len(stats_data))

else:
    st.info("Please upload the shape image to start detection.")
