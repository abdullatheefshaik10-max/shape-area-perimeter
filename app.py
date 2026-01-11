import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("Shape & Contour Analyzer (Stable Version)")
st.write("Precise detection for clean geometric images")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def classify_shape(cnt):
    shape = "Unknown"

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    area = cv2.contourArea(cnt)
    if area < 500:
        return None

    # TRIANGLE
    if vertices == 3:
        shape = "Triangle"

    # QUADRILATERALS
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)

        if 0.90 <= ar <= 1.10:
            shape = "Square / Diamond"
        elif ar > 1.2:
            shape = "Rectangle"
        else:
            shape = "Trapezium"

    # POLYGONS
    elif vertices == 5:
        shape = "Pentagon"

    elif vertices == 6:
        shape = "Hexagon"

    # CIRCLE & STAR
    else:
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        if solidity < 0.85:
            shape = "Star"
        else:
            shape = "Circle"

    return shape

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
        shape = classify_shape(cnt)
        if shape is None:
            continue

        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

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

        results.append([i + 1, shape, round(area, 2), round(peri, 2)])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Shapes")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
                 use_column_width=True)

    with col2:
        st.subheader("Results")
        st.table(
            [["ID", "Shape", "Area", "Perimeter"]] + results
        )

    st.success(f"Total Shapes Detected: {len(results)}")

else:
    st.info("Upload a clean geometric image to begin detection")
