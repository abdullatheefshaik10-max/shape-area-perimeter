import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="Universal Geometry Shape Detector", layout="wide")
st.title("🔶 Universal Geometry Shape Detector")
st.write("Detects all common geometric shapes using contour & feature analysis")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def classify_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    shape = "Unknown"

    # TRIANGLE
    if vertices == 3:
        shape = "Triangle"

    # QUADRILATERALS
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if 0.95 <= aspect_ratio <= 1.05:
            shape = "Square / Diamond"
        elif solidity < 0.9:
            shape = "Trapezium"
        else:
            shape = "Rectangle"

    # POLYGONS
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices == 6:
        shape = "Hexagon"
    elif vertices == 7:
        shape = "Heptagon"
    elif vertices == 8:
        shape = "Octagon"

    # CURVED & COMPLEX SHAPES
    elif vertices > 8:
        if solidity < 0.8:
            shape = "Star / Irregular Shape"
        else:
            shape = "Circle / Ellipse"

    return shape, vertices

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    results = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        shape, vertices = classify_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(
            output,
            shape,
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        results.append([i + 1, shape, vertices, round(area, 2), round(perimeter, 2)])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Shapes")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

    with col2:
        st.subheader("Shape Measurements")
        st.table(
            [["ID", "Shape", "Vertices", "Area", "Perimeter"]] + results
        )

    st.success(f"Total Objects Detected: {len(results)}")

else:
    st.info("Upload an image to detect geometric shapes")
