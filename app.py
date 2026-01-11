import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Shape & Contour Analyzer")
st.write("Detects all geometric shapes including star, hexagon, trapezium, diamond.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def detect_shape(cnt):
    shape = "Unknown"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    vertices = len(approx)

    if vertices == 3:
        shape = "Triangle"

    elif vertices == 4:
        # Check square / rectangle / diamond
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        if 0.95 <= ar <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"

    elif vertices == 5:
        shape = "Pentagon"

    elif vertices == 6:
        shape = "Hexagon"

    elif vertices >= 8:
        # Detect star vs circle
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area)

        if solidity < 0.8:
            shape = "Star"
        else:
            shape = "Circle"

    return shape, vertices

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result = img.copy()
    data = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        shape, vertices = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        cv2.drawContours(result, [cnt], -1, (0,255,0), 2)
        cv2.putText(result, shape, (cx-40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        data.append([i+1, shape, vertices, round(area,2), round(perimeter,2)])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Annotated Image")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 use_column_width=True)

    with col2:
        st.subheader("Detected Shapes Table")
        st.table(
            [["ID", "Shape", "Vertices", "Area", "Perimeter"]] + data
        )

    st.success(f"Total Shapes Detected: {len(data)}")

else:
    st.info("Upload an image to begin shape detection.")
