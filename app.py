Python 3.13.2 (v3.13.2:4f8bb3947cf, Feb  4 2025, 11:51:10) [Clang 15.0.0 (clang-1500.3.9.4)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Geometric Shape Analyzer", layout="centered")

st.title("Geometric Shape Analyzer")
st.caption("Contour-based shape detection using edge detection")

uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

def classify_shape(approx):
    sides = len(approx)
    if sides == 3:
        return "Triangle"
    elif sides == 4:
        return "Rectangle"
    elif sides == 5:
        return "Pentagon"
    elif sides >= 8:
        return "Circle"
    else:
        return "Unknown"

if uploaded_image:
    img = Image.open(uploaded_image)
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(img, use_column_width=True)

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = img_cv.copy()
...     results = []
... 
...     for i, cnt in enumerate(contours):
...         area = cv2.contourArea(cnt)
...         if area < 200:
...             continue
... 
...         peri = cv2.arcLength(cnt, True)
...         approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
...         shape = classify_shape(approx)
... 
...         x, y, w, h = cv2.boundingRect(cnt)
...         cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
...         cv2.putText(annotated, shape, (x, y-5),
...                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
... 
...         results.append({
...             "Object": i + 1,
...             "Shape": shape,
...             "Area": round(area, 2),
...             "Perimeter": round(peri, 2)
...         })
... 
...     st.subheader("Detected Shapes")
...     st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
... 
...     st.subheader("Summary")
...     col1, col2 = st.columns(2)
...     col1.metric("Total Objects", len(results))
...     col2.metric("Unique Shapes", len(set(r["Shape"] for r in results)))
... 
...     if results:
...         st.subheader("Measurements")
...         st.table(results)
...     else:
...         st.warning("No significant shapes detected")
... 
... else:
...     st.info("Upload an image to begin analysis")
