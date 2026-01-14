import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Golf AI Analyzer", layout="wide")
st.title("⛳ AI Golf Swing Analyzer")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")

# 1. Camera View Selection
view_mode = st.sidebar.radio(
    "Camera Angle", 
    ["Down the Line (DTL)", "Face On"],
    index=0,
    help="DTL: Camera behind the golfer. Face On: Camera facing the chest."
)

# 2. Performance Tuning
processing_speed = st.sidebar.select_slider(
    "Processing Speed", 
    options=["Slow (High Accuracy)", "Fast (Preview)"], 
    value="Fast (Preview)"
)
# Frame skipping logic: Process every 3rd frame for speed, or every frame for accuracy
FRAME_SKIP = 3 if processing_speed == "Fast (Preview)" else 1

# 3. Sensitivity Sliders
st.sidebar.subheader("Fault Detection Sensitivity")
head_thresh = st.sidebar.slider("Head Stability Tolerance", 0.01, 0.10, 0.05, help="Lower = stricter")
sway_thresh = st.sidebar.slider("Sway Tolerance", 0.01, 0.10, 0.05, help="Lower = stricter")

# 4. Instructions
st.sidebar.info(
    "💡 **Tip:** For Tempo analysis, use Slow Motion video (120fps or 240fps) if possible."
)

# --- CLASS: TEMPO CALCULATOR ---
class TempoCalculator:
    def __init__(self):
        self.state = "ADDRESS" # States: ADDRESS, BACKSWING, DOWNSWING, FINISHED
        self.start_x = None
        self.top_y = 10000     # Track highest hand point (smallest Y value)
        self.backswing_frames = 0
        self.downswing_frames = 0
        self.ratio = "N/A"

    def update(self, hands_y, hands_x):
        # 1. DETECT START OF SWING
        if self.state == "ADDRESS":
            if self.start_x is None:
                self.start_x = hands_x
            # Trigger: Hands move significantly (20px)
            if abs(hands_x - self.start_x) > 20: 
                self.state = "BACKSWING"

        # 2. TRACK BACKSWING
        elif self.state == "BACKSWING":
            self.backswing_frames += 1
            # Check for top of backswing (lowest Y pixel value is highest point)
            if hands_y < self.top_y:
                self.top_y = hands_y
            # Trigger: Hands drop significantly below the top
            elif hands_y > (self.top_y + 30): 
                self.state = "DOWNSWING"

        # 3. TRACK DOWNSWING
        elif self.state == "DOWNSWING":
            self.downswing_frames += 1
            # Trigger: Hands return to start X (Impact) OR timeout
            if abs(hands_x - self.start_x) < 20 or self.downswing_frames > 20:
                self.state = "FINISHED"
                if self.downswing_frames > 0:
                    r = self.backswing_frames / self.downswing_frames
                    self.ratio = f"{r:.1f}:1"

# --- HELPER: CALCULATE LINE Y ---
def calculate_line_y(p1, p2, x):
    (x1, y1), (x2, y2) = p1, p2
    if x2 - x1 == 0: return 0
    m = (y2 - y1) / (x2 - x1)
    b = y1 - (m * x1)
    return int(m * x + b)

# --- MAIN APPLICATION LOGIC ---
uploaded_file = st.file_uploader("Upload a golf swing video (MP4, MOV)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save temp file for OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    st_frame = st.empty()
    progress_bar = st.progress(0)
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize State Variables
    tempo_calc = TempoCalculator()
    frame_count = 0
    
    # Report Card Metrics
    max_head_deviation = 0.0
    sway_detected = False
    plane_faults = []

    # Calibration Data
    calibrated = False
    calibration_frames = 0
    addr_nose = None
    addr_hip_center = None
    virtual_ball = None
    shoulder_plane_target = None
    hip_plane_target = None

    # --- VIDEO PROCESSING LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Optimization: Skip frames if in Fast Mode
        if frame_count % FRAME_SKIP != 0:
            continue

        # Resize to standard width (640px) for consistent processing speed
        frame = cv2.resize(frame, (640, int(640 * (frame.shape[0] / frame.shape[1]))))
        h, w, _ = frame.shape
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # --- EXTRACT CRITICAL LANDMARKS ---
            nose = (int(lm[0].x * w), int(lm[0].y * h))
            l_hip = (int(lm[23].x * w), int(lm[23].y * h))
            r_hip = (int(lm[24].x * w), int(lm[24].y * h))
            hip_center_x = (l_hip[0] + r_hip[0]) / 2
            
            # Hands & Shoulders
            r_shoulder = (int(lm[12].x * w), int(lm[12].y * h))
            l_wrist = (int(lm[15].x * w), int(lm[15].y * h))
            r_wrist = (int(lm[16].x * w), int(lm[16].y * h))
            hands = ((l_wrist[0] + r_wrist[0]) // 2, (l_wrist[1] + r_wrist[1]) // 2)

            # --- UPDATE TEMPO CALCULATOR ---
            tempo_calc.update(hands[1], hands[0])

            # --- PHASE 1: CALIBRATION ---
            if not calibrated:
                calibration_frames += 1
                cv2.putText(frame, "CALIBRATING...", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Calibrate quickly (first 5-10 processed frames)
                if calibration_frames > (5 / FRAME_SKIP): 
                    addr_nose = nose
                    addr_hip_center = hip_center_x
                    
                    # DTL Swing Plane Setup
                    if view_mode == "Down the Line (DTL)":
                        if hands[0] != r_shoulder[0]:
                            # Calculate arm slope to find virtual ball
                            m = (hands[1] - r_shoulder[1]) / (hands[0] - r_shoulder[0])
                            b = r_shoulder[1] - (m * r_shoulder[0])
                            ball_y = h
                            ball_x = int((ball_y - b) / m)
                            virtual_ball = (ball_x, ball_y)
                            shoulder_plane_target = r_shoulder
                            hip_plane_target = r_hip
                    calibrated = True
            
            # --- PHASE 2: ANALYSIS ---
            else:
                # A. HEAD STABILITY
                diff_y = abs(nose[1] - addr_nose[1])
                deviation_pct = diff_y / h
                if deviation_pct > max_head_deviation:
                    max_head_deviation = deviation_pct
                
                color_head = (0, 255, 0) if deviation_pct < head_thresh else (0, 0, 255)
                cv2.circle(frame, addr_nose, 20, color_head, 2)

                # B. SWAY DETECTION
                box_left = int(addr_hip_center - (w * sway_thresh))
                box_right = int(addr_hip_center + (w * sway_thresh))
                
                if hip_center_x < box_left or hip_center_x > box_right:
                    sway_detected = True
                    color_sway = (0, 0, 255)
                else:
                    color_sway = (0, 255, 0)
                
                cv2.line(frame, (box_left, 0), (box_left, h), color_sway, 2)
                cv2.line(frame, (box_right, 0), (box_right, h), color_sway, 2)

                # C. SWING PLANE (DTL Only)
                if view_mode == "Down the Line (DTL)" and virtual_ball:
                    cv2.line(frame, virtual_ball, shoulder_plane_target, (255, 255, 0), 2)
                    cv2.line(frame, virtual_ball, hip_plane_target, (255, 255, 0), 2)
                    
                    upper_limit = calculate_line_y(virtual_ball, shoulder_plane_target, hands[0])
                    lower_limit = calculate_line_y(virtual_ball, hip_plane_target, hands[0])

                    if hands[1] < upper_limit:
                        cv2.putText(frame, "OVER THE TOP", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        plane_faults.append("Over the Top")
                    elif hands[1] > lower_limit:
                        cv2.putText(frame, "TOO SHALLOW", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        plane_faults.append("Too Shallow")

                # D. DISPLAY TEMPO
                cv2.putText(frame, f"Tempo: {tempo_calc.ratio}", (w - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update Streamlit Display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame, channels="RGB", use_container_width=True)
        
        # Update Progress Bar
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    os.remove(tfile.name)
    
    # --- FINAL REPORT CARD ---
    st.divider()
    st.header("📊 Swing Analysis Report")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Head Stability Score
    head_status = "✅ Stable" if max_head_deviation < head_thresh else "❌ Unstable"
    col1.metric("Head Stability", head_status, f"Dev: {int(max_head_deviation*100)}%")

    # 2. Sway Score
    sway_status = "❌ Detected" if sway_detected else "✅ Good Rotation"
    col2.metric("Hip Sway", sway_status)

    # 3. Swing Plane Score
    if view_mode == "Down the Line (DTL)":
        if len(plane_faults) > 5:
            fault = max(set(plane_faults), key=plane_faults.count)
            plane_status = f"⚠️ {fault}"
        else:
            plane_status = "✅ On Plane"
        col3.metric("Swing Path", plane_status)
    else:
        col3.metric("Swing Path", "N/A (Face On)")

    # 4. Tempo Score
    tempo_val = tempo_calc.ratio
    tempo_status = "Waiting..."
    if tempo_val != "N/A":
        try:
            ratio_num = float(tempo_val.split(":")[0])
            if 2.5 <= ratio_num <= 3.5:
                tempo_status = "✅ Tour Rhythm"
            elif ratio_num < 2.5:
                tempo_status = "⚡ Too Fast"
            else:
                tempo_status = "🐢 Too Slow"
        except:
            tempo_status = "❓ Calc Error"
            
    col4.metric("Tempo (Ratio)", tempo_status, tempo_val)