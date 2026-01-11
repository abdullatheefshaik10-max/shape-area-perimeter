import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔷 Advanced Shape & Contour Analyzer")
st.write("Accurately detects: Square, Rectangle, Diamond, Circle, Triangle, Trapezium, Pentagon, Hexagon, Stars")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Sidebar controls
st.sidebar.header("Detection Settings")
threshold_value = st.sidebar.slider("Threshold", 50, 250, 127, 5)
epsilon_factor = st.sidebar.slider("Contour Approximation", 0.01, 0.08, 0.04, 0.01)
min_area = st.sidebar.slider("Minimum Area", 100, 5000, 800, 100)

def angle_between_points(p1, p2, p3):
    """Calculate angle at point p2"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    
    if v1_mag == 0 or v2_mag == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (v1_mag * v2_mag)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))

def detect_shape(cnt):
    """Advanced shape detection algorithm"""
    shape = "Unknown"
    
    # Calculate perimeter and approximate the contour
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon_factor * peri, True)
    vertices = len(approx)
    
    # Calculate geometric properties
    area = cv2.contourArea(cnt)
    
    # Convex hull for solidity calculation
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Circularity
    circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)
    
    # Rotated rectangle for detecting rotation
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (width, height), rotation = rect
    
    # Normalize rotation angle
    if width < height:
        rotation = rotation + 90
    
    # ===== SHAPE DETECTION LOGIC =====
    
    # TRIANGLE
    if vertices == 3:
        shape = "Triangle"
    
    # QUADRILATERALS (4 vertices)
    elif vertices == 4:
        # Extract corner points
        points = [approx[i][0] for i in range(4)]
        
        # Calculate all side lengths
        sides = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            side_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            sides.append(side_length)
        
        # Calculate all interior angles
        angles = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            angle = angle_between_points(p1, p2, p3)
            angles.append(angle)
        
        # Side length statistics
        side_mean = np.mean(sides)
        side_std = np.std(sides)
        side_variance_coeff = side_std / side_mean if side_mean > 0 else 1
        
        # Check if all sides are roughly equal
        sides_equal = side_variance_coeff < 0.20
        
        # Check if all angles are close to 90 degrees
        angles_perpendicular = all(75 < angle < 105 for angle in angles)
        
        # DIAMOND: Equal sides + perpendicular angles + rotated 35-55 degrees
        if sides_equal and angles_perpendicular:
            # Check rotation
            abs_rotation = abs(rotation)
            is_rotated = (35 < abs_rotation < 55) or (35 < abs(abs_rotation - 90) < 55)
            
            if is_rotated:
                shape = "Diamond"
            elif 0.90 <= aspect_ratio <= 1.10:
                shape = "Square"
            else:
                shape = "Rectangle"
        
        # TRAPEZIUM: Not all angles are 90 degrees, check for parallel sides
        elif not angles_perpendicular:
            # A trapezium has at least one pair of parallel sides
            # Check opposite angle pairs
            opposite_sum_1 = angles[0] + angles[2]
            opposite_sum_2 = angles[1] + angles[3]
            
            # If opposite angles sum to roughly 180, we have parallel sides
            has_parallel = (170 < opposite_sum_1 < 190) or (170 < opposite_sum_2 < 190)
            
            if has_parallel:
                shape = "Trapezium"
            else:
                shape = "Quadrilateral"
        
        # RECTANGLE or SQUARE
        else:
            if 0.90 <= aspect_ratio <= 1.10 and sides_equal:
                shape = "Square"
            else:
                shape = "Rectangle"
    
    # PENTAGON
    elif vertices == 5:
        # Check if it's a star based on solidity
        if solidity < 0.65:
            shape = "Star (5-point)"
        else:
            shape = "Pentagon"
    
    # HEXAGON
    elif vertices == 6:
        # Check if it's a 6-point star
        if solidity < 0.65:
            shape = "Star (6-point)"
        else:
            shape = "Hexagon"
    
    # 7-10 vertices
    elif 7 <= vertices <= 10:
        if solidity < 0.70:
            shape = f"Star ({vertices}-point)"
        else:
            shape = f"Polygon ({vertices} sides)"
    
    # CIRCLE or STAR (many vertices)
    elif vertices > 10:
        if circularity > 0.70:
            shape = "Circle"
        elif solidity < 0.75:
            shape = "Star"
        else:
            shape = "Circle"
    
    return shape, vertices, solidity, circularity

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    img = np.array(image)
    
    # Handle different color formats
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img.copy()
    data = []
    
    # Sort contours by position (top to bottom, left to right)
    def get_position(cnt):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
        return (cy // 150, cx)  # Group by rows, then columns
    
    contours = sorted(contours, key=get_position)
    
    # Process each contour
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Skip small contours
        if area < min_area:
            continue
        
        shape, vertices, solidity, circularity = detect_shape(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
        
        # Color coding for different shapes
        color_map = {
            "Square": (0, 255, 0),       # Green
            "Rectangle": (255, 165, 0),  # Orange
            "Diamond": (255, 0, 255),    # Magenta
            "Circle": (0, 255, 255),     # Cyan
            "Triangle": (255, 255, 0),   # Yellow
            "Hexagon": (0, 128, 255),    # Light blue
            "Trapezium": (128, 0, 255),  # Purple
            "Pentagon": (255, 128, 0),   # Orange-yellow
        }
        
        # Get color (default green)
        for key in color_map:
            if key in shape:
                color = color_map[key]
                break
        else:
            color = (0, 255, 0)
        
        # Draw contour
        cv2.drawContours(result, [cnt], -1, color, 3)
        
        # Add text label with background
        label = shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Background rectangle for text
        text_x = cx - text_size[0] // 2
        text_y = cy - 25
        cv2.rectangle(result, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(result, label, (text_x, text_y),
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Draw ID number
        cv2.putText(result, f"#{i+1}", (cx - 15, cy + 10),
                   font, 0.5, (0, 0, 255), 2)
        
        # Store data
        data.append([
            i + 1,
            shape,
            vertices,
            round(area, 1),
            round(perimeter, 1),
            round(solidity, 3),
            round(circularity, 3)
        ])
    
    # Display results
    col1, col2 = st.columns([1.4, 1])
    
    with col1:
        st.subheader("🎯 Detected Shapes")
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    with col2:
        st.subheader("📊 Shape Analysis")
        if data:
            import pandas as pd
            df = pd.DataFrame(data, columns=[
                "ID", "Shape", "Vertices", "Area (px²)", 
                "Perimeter (px)", "Solidity", "Circularity"
            ])
            st.dataframe(df, use_container_width=True, height=400)
            
            st.success(f"✅ Detected {len(data)} shapes")
            
            # Shape distribution
            st.subheader("📈 Distribution")
            shape_counts = {}
            for row in data:
                shape_name = row[1]
                shape_counts[shape_name] = shape_counts.get(shape_name, 0) + 1
            
            for shape_name, count in sorted(shape_counts.items()):
                st.write(f"**{shape_name}**: {count}")
        else:
            st.warning("⚠️ No shapes detected. Adjust settings in sidebar.")
    
    # Debug preprocessing view
    with st.expander("🔍 View Preprocessing Steps"):
        debug_col1, debug_col2, debug_col3 = st.columns(3)
        
        with debug_col1:
            st.write("**1. Grayscale**")
            st.image(gray, use_column_width=True, clamp=True)
        
        with debug_col2:
            st.write("**2. Blurred**")
            st.image(blur, use_column_width=True, clamp=True)
        
        with debug_col3:
            st.write("**3. Threshold**")
            st.image(thresh, use_column_width=True, clamp=True)

else:
    st.info("📤 **Upload an image to start detecting shapes**")
    
    st.markdown("""
    ### 🎯 Detectable Shapes:
    
    | Shape | Detection Method |
    |-------|------------------|
    | ⬜ **Square** | 4 equal sides, 90° angles, not rotated |
    | 🔲 **Rectangle** | 4 sides, 90° angles, unequal sides |
    | 🔷 **Diamond** | 4 equal sides, 90° angles, rotated 45° |
    | ⭕ **Circle** | High circularity (>0.7), many vertices |
    | 🔺 **Triangle** | 3 vertices |
    | ⬢ **Hexagon** | 6 vertices, high solidity |
    | 🔶 **Trapezium** | 4 sides, one pair parallel |
    | ⭐ **Star** | Low solidity (<0.65), concave shape |
    | ⬟ **Pentagon** | 5 vertices |
    
    ### ⚙️ Settings Guide:
    
    **Threshold** (50-250): Controls edge detection
    - Lower values: Detect lighter shapes
    - Higher values: Detect darker shapes
    - Try 100-150 for white backgrounds
    
    **Contour Approximation** (0.01-0.08): Shape precision
    - Lower values: More precise, more vertices
    - Higher values: Simpler shapes, fewer vertices
    - Recommended: 0.03-0.05
    
    **Minimum Area** (100-5000): Filter small noise
    - Increase to ignore small artifacts
    - Decrease to detect smaller shapes
    """)
