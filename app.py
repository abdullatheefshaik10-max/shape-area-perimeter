import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="Robust Shape Analyzer", layout="wide")
st.title("🔷 Robust Shape & Contour Analyzer")
st.write("Rule-based geometric shape classification using contour features")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def angle(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    return math.degrees(math.acos((a*a + c*c - b*b) / (2*a*c + 1e-10)))

def classify_shape(cnt):
    shape = "Irregular"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)
    area = cv2.contourArea(cnt)

    if area < 400:
        return None

    # CIRCULARITY
    circularity = 4 * math.pi * area / (peri * peri + 1e-10)

    # TRIANGLE
    if vertices == 3:
        shape = "Triangle"

    # QUADRILATERALS
    elif vertices == 4:
        pts = approx.reshape(4, 2)
        angles = []
        for i in range(4):
            angles.append(angle(pts[i], pts[(i+1)%4], pts[(i+2)%4]))

        if all(80 <= a <= 100 for a in angles):
            x, y, w, h = cv2.boundingRect(pts)
            ar = w / float(h)
            if 0.95 <= ar <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        else:
            shape = "Trapezium / Diamond"

    # POLYGONS
    elif 5 <= vertices <= 8:
        shape = f"{vertices}-sided Polygon"

    # CIRCLE / STAR / COMPLEX
    else:
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        if circularity > 0.75:
            shape = "Circle"
        elif solidity < 0.85:
            shape = "Star / Concave Shape"
        else:
            shape = "Ellipse / Curve"

    return shape

if uploaded_file:
    image = Image.open(uploaded_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    results = []

    for i, cnt in enumerate(contours):
        shape = classify_shape(cnt)
        if shape is None:
            continue

        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(output, shape, (cx-40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        results.append([i+1, shape, round(area,2), round(peri,2)])

    col1, col2 = st.columns(2)

    with col1:
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.table([["ID","Shape","Area","Perimeter"]] + results)

    st.success(f"Objects Detected: {len(results)}")

else:
    st.info("Upload an image to analyze shapes")
