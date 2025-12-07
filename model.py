import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time

# ------------------- Vegetation Indices -------------------
def compute_indices(bgr):
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)

    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 1, cv2.NORM_MINMAX)

    vari = (g - r) / (g + r - b + 1e-6)
    vari = cv2.normalize(vari, None, 0, 1, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) / 179.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    return exg, vari, hue, sat, val


# ------------------- Stress Heatmap -------------------
def stress_score(exg, vari, hue, sat, val, veg_mask, w_exg, w_var, w_y, w_sat):
    s_exg = 1.0 - exg
    s_var = 1.0 - vari
    y_prox = np.clip(1.0 - np.abs(hue - 0.145) / 0.035, 0, 1)  # Yellow proximity
    s_sat = 1.0 - sat

    score = w_exg * s_exg + w_var * s_var + w_y * y_prox + w_sat * s_sat
    score = np.clip(score, 0, 1)
    score = score * (veg_mask.astype(bool))
    return cv2.GaussianBlur(score, (0, 0), 1.2)


# ------------------- YOLO Annotations (Gray Background + Yellow Stress) -------------------
def create_yolo_annotations_from_heatmap(score, stress_cutoff, veg_mask, img, min_area=1000):
    h, w, _ = img.shape
    mask = ((score > stress_cutoff) & (veg_mask.astype(bool))).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert original to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    annotated = gray_bgr.copy()
    yolo_annotations = []

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            # Fill the stress region with yellow
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 255), -1)
            # Outline in red for clarity
            cv2.drawContours(annotated, [cnt], -1, (0, 0, 255), 2)

            # Save YOLO-style polygon
            points = cnt.reshape(-1, 2)
            norm_points = [(p[0] / w, p[1] / h) for p in points]
            yolo_annotations.append(norm_points)

    return annotated, yolo_annotations, mask


# ------------------- Streamlit App -------------------
def main():
    st.title("🌱 Plant Stress Detection with Heatmap + YOLO Annotations")

    # Sidebar sliders
    st.sidebar.header("Thresholds & Weights")
    exg_thr = st.sidebar.slider("ExG Threshold", 0, 100, 45) / 100.0
    vari_thr = st.sidebar.slider("VARI Threshold", 0, 100, 50) / 100.0
    stress_cutoff = st.sidebar.slider("Stress Cutoff", 0, 100, 50) / 100.0
    min_area = st.sidebar.slider("Minimum Area (px)", 100, 10000, 2000)

    w_exg = st.sidebar.slider("Weight ExG", 0, 100, 35) / 100.0
    w_var = st.sidebar.slider("Weight VARI", 0, 100, 30) / 100.0
    w_y = st.sidebar.slider("Weight Yellow", 0, 100, 25) / 100.0
    w_sat = st.sidebar.slider("Weight Sat", 0, 100, 10) / 100.0

    # Create tabs
    tab1, tab2 = st.tabs(["📂 Upload Image", "📹 Live Camera Feed"])

    # --- Tab 1: Upload Image ---
    with tab1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Load image
            pil_img = Image.open(uploaded_file).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1].copy()  # BGR

            # Indices
            exg, vari, hue, sat, val = compute_indices(img)

            # Vegetation mask
            veg = (exg > exg_thr) & (vari > vari_thr) & (val > 0.1)
            veg = veg.astype(np.uint8)

            # Stress score
            score = stress_score(exg, vari, hue, sat, val, veg, w_exg, w_var, w_y, w_sat)

            # Heatmap overlay
            heat = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat[veg == 0] = 0
            overlay = cv2.addWeighted(img, 0.6, heat, 0.6, 0)

            # YOLO Annotations with grayscale background
            annotated, yolo_annotations, mask = create_yolo_annotations_from_heatmap(
                score, stress_cutoff, veg, img, min_area
            )

            # --- Display results ---
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

            st.subheader("Heatmap Overlay")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Stress Heatmap", use_container_width=True)

            st.subheader("Annotated Stress Regions (Grayscale Background)")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="YOLO Annotations", use_container_width=True)

            if yolo_annotations:
                st.subheader("YOLO Segmentation Annotations")
                st.json(yolo_annotations)

    # --- Tab 2: Live Camera Feed ---
    with tab2:
        st.write("📹 **Live Camera Feed with Real-time Stress Detection**")
        
        # Control buttons
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            start_camera = st.button("▶️ Start")
        with col_btn2:
            stop_camera = st.button("⏹️ Stop")
        
        # Initialize session state
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        if start_camera:
            st.session_state.camera_running = True
        if stop_camera:
            st.session_state.camera_running = False
        
        if st.session_state.camera_running:
            stframe1 = st.empty()
            stframe2 = st.empty()
            stframe3 = st.empty()
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access camera. Please check your camera.")
                st.session_state.camera_running = False
            else:
                st.success("✅ Camera active - Processing frames...")
                
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Process frame
                    img = frame.copy()
                    exg, vari, hue, sat, val = compute_indices(img)
                    veg = (exg > exg_thr) & (vari > vari_thr) & (val > 0.1)
                    veg = veg.astype(np.uint8)
                    score = stress_score(exg, vari, hue, sat, val, veg, w_exg, w_var, w_y, w_sat)
                    
                    # Heatmap overlay
                    heat = cv2.applyColorMap((score * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    heat[veg == 0] = 0
                    overlay = cv2.addWeighted(img, 0.6, heat, 0.6, 0)
                    
                    # YOLO Annotations
                    annotated, _, _ = create_yolo_annotations_from_heatmap(
                        score, stress_cutoff, veg, img, min_area
                    )
                    
                    # Display in three columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        stframe1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                                      caption="🎥 Live Feed", 
                                      channels="RGB",
                                      use_container_width=True)
                    with col2:
                        stframe2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                                      caption="🔥 Heatmap", 
                                      channels="RGB",
                                      use_container_width=True)
                    with col3:
                        stframe3.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                                      caption="📍 Annotations", 
                                      channels="RGB",
                                      use_container_width=True)
                    
                    time.sleep(0.03)  # ~30 FPS
                
                cap.release()
        else:
            st.info("💡 Click 'Start' to begin live camera feed with stress detection")


if __name__ == "__main__":
    main()