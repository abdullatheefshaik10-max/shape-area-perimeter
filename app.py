import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Advanced Shape & Contour Analyzer")
st.write("Accurately detects geometric shapes: square, circle, star, triangle, trapezium, diamond, rectangle, hexagon")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Add threshold slider for fine-tuning
threshold_val = st.sidebar.slider("Threshold Value", 50, 250, 127, 5)
min_area = st.sidebar.slider("Minimum Shape Area", 100, 5000, 1000, 100)

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if magnitude == 0:
        return 0
    
    cos_angle = dot_product / magnitude
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)

def detect_shape(cnt):
    """Enhanced shape detection with multiple algorithms"""
    shape = "Unknown"
    
    # Calculate contour properties
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    
    # Use different epsilon values for approximation
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    vertices = len(approx)
    
    # Calculate shape metrics
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Circularity (4π*Area/Perimeter²)
    circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
    
    # Bounding box properties
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    
    # Minimum area rectangle (can detect rotation)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    rect_area = rect[1][0] * rect[1][1]
    extent = area / rect_area if rect_area > 0 else 0
    
    # Rotation angle from minimum area rectangle
    rotation_angle = rect[2]
    
    # TRIANGLE - 3 vertices
    if vertices == 3:
        shape = "Triangle"
    
    # QUADRILATERALS - 4 vertices
    elif vertices == 4:
        # Get all side lengths
        sides = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i+1)%4][0]
            side_len = np.linalg.norm(p1 - p2)
            sides.append(side_len)
        
        # Get all angles
        angles = []
        for i in range(4):
            p1 = approx[i][0]
            p2 = approx[(i+1)%4][0]
            p3 = approx[(i+2)%4][0]
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)
        
        # Check if sides are roughly equal
        side_variance = np.std(sides) / np.mean(sides) if np.mean(sides) > 0 else 1
        sides_equal = side_variance < 0.15
        
        # Check if all angles are roughly 90 degrees
        angles_right = all(80 < angle < 100 for angle in angles)
        
        # DIAMOND: Equal sides, right angles, rotated 30-60 degrees
        if sides_equal and angles_right:
            if 30 < abs(rotation_angle) < 60 or 30 < abs(rotation_angle - 90) < 60:
                shape = "Diamond"
            elif 0.90 <= aspect_ratio <= 1.10:
                shape = "Square"
            else:
                shape = "Rectangle"
        
        # TRAPEZIUM: One pair of parallel sides (two angles sum to ~180)
        elif not angles_right:
            # Check for parallel sides by angle analysis
            angle_pairs = [
                (angles[0] + angles[2], angles[1] + angles[3]),
                (angles[0] + angles[1], angles[2] + angles[3])
            ]
            
            has_parallel = any(170 < sum_pair < 190 for pair in angle_pairs for sum_pair in pair)
            
            if has_parallel:
                shape = "Trapezium"
            else:
                shape = "Rectangle"
        
        # RECTANGLE/SQUARE: Right angles, different side lengths
        else:
            if 0.90 <= aspect_ratio <= 1.10 and sides_equal:
                shape = "Square"
            else:
                shape = "Rectangle"
    
    # PENTAGON - 5 vertices
    elif vertices == 5:
        if solidity < 0.70:
            shape = "Star (5-point)"
        else:
            shape = "Pentagon"
    
    # HEXAGON - 6 vertices
    elif vertices == 6:
        if solidity < 0.70:
            shape = "Star (6-point)"
        else:
            shape = "Hexagon"
    
    # 7-8 vertices
    elif vertices == 7 or vertices == 8:
        if solidity < 0.75:
            shape = f"Star ({vertices}-point)"
        else:
            shape = f"{vertices}-gon"
    
    # CIRCLE or complex star - more than 8 vertices
    elif vertices > 8:
        if circularity > 0.75:
            shape = "Circle"
        elif solidity < 0.75:
            shape = "Star"
        else:
            shape = "Circle"
    
    return shape, vertices, solidity, circularity, aspect_ratio

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Handle different image formats
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    _, thresh = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    data = []
    
    # Sort contours by Y position then X position (top to bottom, left to right)
    def get_contour_position(cnt):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
        # Group by rows (every 200 pixels), then by x position
        return (cy // 200, cx)
    
    contours = sorted(contours, key=get_contour_position)
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Filter out very small contours
        if area < min_area:
            continue
        
        shape, vertices, solidity, circularity, aspect_ratio = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
        
        # Draw contours with different colors for different shapes
        colors = {
            "Square": (0, 255, 0),
            "Rectangle": (255, 165, 0),
            "Diamond": (255, 0, 255),
            "Circle": (0, 255, 255),
            "Triangle": (255, 255, 0),
            "Hexagon": (0, 128, 255),
            "Trapezium": (128, 0, 255),
        }
        
        color = colors.get(shape, (0, 255, 0))
        cv2.drawContours(result, [cnt], -1, color, 3)
        
        # Add shape label with background
        label = f"{shape}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (cx - label_size[0]//2 - 5, cy - 35), 
                     (cx + label_size[0]//2 + 5, cy - 10), (255, 255, 255), -1)
        cv2.putText(result, label, (cx - label_size[0]//2, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add ID number
        cv2.putText(result, f"#{i+1}", (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        data.append([
            i + 1,
            shape,
            vertices,
            round(area, 1),
            round(perimeter, 1),
            round(solidity, 3),
            round(circularity, 3),
            round(aspect_ratio, 2)
        ])
    
    # Display results
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.subheader("Annotated Image")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col2:
        st.subheader("Detected Shapes")
        if data:
            import pandas as pd
            df = pd.DataFrame(data, columns=[
                "ID", "Shape", "Vertices", "Area", 
                "Perimeter", "Solidity", "Circularity", "Aspect Ratio"
            ])
            st.dataframe(df, use_container_width=True, height=400)
            
            st.success(f"✅ Total: {len(data)} shapes")
            
            # Shape counts
            st.subheader("Shape Distribution")
            shape_counts = {}
            for row in data:
                shape = row[1]
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            
            for shape, count in sorted(shape_counts.items()):
                st.write(f"**{shape}**: {count}")
        else:
            st.warning("⚠️ No shapes detected. Adjust threshold slider.")
    
    # Debug view
    with st.expander("🔍 Preprocessing & Debug"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Grayscale**")
            st.image(gray, use_column_width=True, clamp=True)
        with col_b:
            st.write("**Blurred**")
            st.image(blur, use_column_width=True, clamp=True)
        with col_c:
            st.write("**Threshold**")
            st.image(thresh, use_column_width=True, clamp=True)

else:
    st.info("📤 Upload an image to begin shape detection")
    
    st.markdown("""
    ### Supported Shapes:
    - ⬜ **Square** - Equal sides, 90° angles, not rotated
    - 🔲 **Rectangle** - 90° angles, different length sides
    - 🔷 **Diamond** - Equal sides, 90° angles, rotated 45°
    - ⭕ **Circle** - High circularity, many vertices
    - 🔺 **Triangle** - 3 vertices
    - ⬢ **Hexagon** - 6 vertices, high solidity
    - ⭐ **Star** - Low solidity (concave shape)
    - 🔶 **Trapezium** - 4 sides with one pair parallel
    
    ### Adjust Settings:
    Use the **sidebar sliders** to fine-tune detection:
    - **Threshold**: Adjust edge detection sensitivity
    - **Min Area**: Filter out small noise/artifacts
    """)
