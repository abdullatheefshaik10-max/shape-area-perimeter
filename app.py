import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Advanced Shape & Contour Analyzer")
st.write("Accurately detects geometric shapes: square, circle, star, triangle, trapezium, diamond, rectangle, hexagon")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def detect_shape(cnt):
    """Enhanced shape detection with better accuracy"""
    shape = "Unknown"
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    vertices = len(approx)
    
    # Get bounding box properties
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    
    # Calculate shape metrics
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / float(hull_area) if hull_area > 0 else 0
    
    # Calculate circularity
    circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    
    if vertices == 3:
        shape = "Triangle"
    
    elif vertices == 4:
        # Detect Square, Rectangle, Diamond, or Trapezium
        # Get the angles between sides
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i+1)%4][0]
            p3 = approx[(i+2)%4][0]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
            angles.append(np.degrees(angle))
        
        # Check if it's a diamond (rotated square)
        # Diamond has all sides roughly equal and is rotated ~45 degrees
        rotation_angle = cv2.minAreaRect(cnt)[2]
        
        # Calculate side lengths
        sides = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i+1)%4][0]
            side_len = np.linalg.norm(p1 - p2)
            sides.append(side_len)
        
        sides_equal = max(sides) / (min(sides) + 1e-6) < 1.3
        angles_right = all(80 < a < 100 for a in angles)
        
        # Diamond detection: rotated square
        if sides_equal and angles_right and (30 < abs(rotation_angle) < 60 or abs(rotation_angle - 45) < 15):
            shape = "Diamond"
        
        # Square: equal sides, right angles, not rotated
        elif 0.95 <= aspect_ratio <= 1.05 and sides_equal and angles_right:
            shape = "Square"
        
        # Trapezium: exactly one pair of parallel sides
        elif not all(80 < a < 100 for a in angles):
            # Check if opposite sides are parallel
            parallel_count = sum(1 for i in range(2) if abs(angles[i] + angles[i+2] - 180) < 20)
            if parallel_count >= 1:
                shape = "Trapezium"
            else:
                shape = "Rectangle"
        
        else:
            shape = "Rectangle"
    
    elif vertices == 5:
        # Could be pentagon or 5-pointed star
        if solidity < 0.65:
            shape = "Star (5-point)"
        else:
            shape = "Pentagon"
    
    elif vertices == 6:
        # Could be hexagon or 6-pointed star
        if solidity < 0.65:
            shape = "Star (6-point)"
        else:
            shape = "Hexagon"
    
    elif vertices > 8:
        # Circle or star with many points
        if circularity > 0.8:
            shape = "Circle"
        elif solidity < 0.75:
            shape = f"Star ({vertices}-point)"
        else:
            shape = "Circle"
    
    elif vertices >= 7:
        # 7-8 sided shapes
        if solidity < 0.65:
            shape = f"Star ({vertices}-point)"
        else:
            shape = f"{vertices}-gon"
    
    return shape, vertices, solidity, circularity

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Handle grayscale images
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding for better edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try multiple threshold methods for better detection
    _, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_or(thresh1, thresh2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    data = []
    
    # Sort contours by position (top to bottom, left to right)
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1] // 100, cv2.boundingRect(c)[0]))
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter out very small contours
        if area < 1000:
            continue
        
        shape, vertices, solidity, circularity = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        # Draw contours and labels
        color = (0, 255, 0)
        cv2.drawContours(result, [cnt], -1, color, 3)
        
        # Add shape label
        cv2.putText(result, shape, (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add ID number
        cv2.putText(result, f"#{i+1}", (cx - 20, cy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        data.append([
            i + 1,
            shape,
            vertices,
            round(area, 2),
            round(perimeter, 2),
            round(solidity, 3),
            round(circularity, 3)
        ])
    
    # Display results
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("Annotated Image")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col2:
        st.subheader("Detected Shapes")
        if data:
            import pandas as pd
            df = pd.DataFrame(data, columns=[
                "ID", "Shape", "Vertices", "Area (px²)", 
                "Perimeter (px)", "Solidity", "Circularity"
            ])
            st.dataframe(df, use_container_width=True)
            
            st.success(f"✅ Total Shapes Detected: {len(data)}")
            
            # Shape distribution
            st.subheader("Shape Distribution")
            shape_counts = {}
            for row in data:
                shape = row[1]
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
            for shape, count in sorted(shape_counts.items()):
                st.write(f"**{shape}**: {count}")
        else:
            st.warning("No shapes detected. Try adjusting the image or threshold settings.")
    
    # Add threshold visualization
    with st.expander("🔍 View Preprocessing Steps"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Original Grayscale**")
            st.image(gray, use_column_width=True, clamp=True)
        with col_b:
            st.write("**Blurred**")
            st.image(blur, use_column_width=True, clamp=True)
        with col_c:
            st.write("**Threshold**")
            st.image(thresh, use_column_width=True, clamp=True)

else:
    st.info("📤 Upload an image to begin shape detection.")
    
    st.markdown("""
    ### Supported Shapes:
    - ⬜ Square
    - 🔲 Rectangle  
    - 🔷 Diamond (rotated square)
    - ⭕ Circle
    - 🔺 Triangle
    - ⬢ Hexagon
    - ⭐ Star (5-point, 6-point, etc.)
    - 🔶 Trapezium
    - And more polygons!
    
    ### Tips for Best Results:
    - Use images with clear, solid shapes on contrasting backgrounds
    - Ensure shapes are not overlapping
    - Higher resolution images work better
    """)
