import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Final Shape Detector", layout="wide")
st.title("🔷 Precision Shape & Contour Analyzer")

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
        if extent > 0.88:
            return "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif 0.65 < extent <= 0.88:
            return "Trapezium"
        else:
            return "Diamond"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Star" if solidity < 0.7 else "Hexagon"
    elif vertices > 6:
        return "Star" if solidity < 0.7 else "Circle"
    return "Unknown"

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    total_img_area = img.shape[0] * img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)

    # Use RETR_CCOMP to get a 2-level hierarchy (External vs Internal)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    stats_data = []
    actual_count = 0

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            # HIERARCHY FIX:
            # hierarchy[i][3] == -1 means the contour has NO parent (it is the outermost).
            # In your image, the outermost contour is the white background box.
            # We want the shapes INSIDE that box. These shapes will have a parent.
            has_parent = hierarchy[i][3] != -1
            
            # Ignore tiny noise AND ignore the outermost background container
            if area < 500 or not has_parent or area > (total_img_area * 0.8):
                continue

            actual_count += 1
            shape = detect_shape_refined(cnt)
            peri = cv2.arcLength(cnt, True)
            
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

            cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 3)
            cv2.putText(result_img, shape, (cx - 40, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            stats_data.append({
                "ID": actual_count,
                "Detected Shape": shape,
                "Area": round(area, 1),
                "Perimeter": round(peri, 1)
            })

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
