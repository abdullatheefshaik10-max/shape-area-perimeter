import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("Shape & Contour Analyzer")
st.write("Accurate detection of all geometric shapes including stars")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def classify_shape(cnt):
    shape = "Unknown"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
    vertices = len(approx)

    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0

    if vertices == 3:
        shape = "Triangle"

    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)

        if 0.90 <= ar <= 1.10:
            shape = "Square"
        else:
            shape = "Rectangle"

    elif vertices == 5:
        shape = "Pentagon"

    elif vertices == 6:
        shape = "Hexagon"

    elif vertices > 6:
        if solidity < 0.75:
            shape = "Star"
        else:
            shape = "Circle"

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
        if area < 400:
            continue

        shape, vertices = classify_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(
            output,
            shape,
            (cx - 30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        results.append([
            i + 1,
            shape,
            vertices,
            round(area, 2),
            round(perimeter, 2)
        ])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Shapes")
        st.image(
            cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

    with col2:
        st.subheader("Shape Details")
        st.table(
            [["ID", "Shape", "Vertices", "Area", "Perimeter"]] + results
        )

    st.success(f"Total Shapes Detected: {len(results)}")

else:
    st.info("Upload the image to detect shapes.")
