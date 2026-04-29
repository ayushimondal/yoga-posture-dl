import tensorflow as tf
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import queue
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(
    page_title="YogaVision",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #09090b; /* Deep zinc background */
    color: #fafafa;
    font-family: 'Inter', sans-serif;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.08), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(217, 70, 239, 0.05), transparent 25%);
    background-color: #09090b;
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Hero header */
.hero {
    padding: 32px 56px 24px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    background: linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 100%);
}
.hero-left { display: flex; flex-direction: column; gap: 4px; }
.hero-tag {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.0;
    letter-spacing: -1px;
}
.hero-title em {
    font-style: normal;
    color: #a78bfa;
}
.hero-sub {
    font-size: 15px;
    color: rgba(255,255,255,0.5);
    font-weight: 400;
    margin-top: 6px;
    letter-spacing: 0.2px;
}
.hero-stat {
    text-align: right;
}
.hero-stat-num {
    font-family: 'Outfit', sans-serif;
    font-size: 42px;
    font-weight: 600;
    color: #a78bfa;
    line-height: 1;
}
.hero-stat-label {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
    font-weight: 500;
}

/* Main layout */
.main-layout {
    display: grid;
    grid-template-columns: 1fr 360px;
    gap: 0;
    height: calc(100vh - 140px);
}

/* Side panel */
.side-panel {
    border-left: 1px solid rgba(255,255,255,0.06);
    padding: 32px 32px 32px 28px;
    display: flex;
    flex-direction: column;
    gap: 28px;
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.01);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

/* Section labels */
.section-label {
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5);
    margin-bottom: 12px;
    font-weight: 600;
}

/* Pose card - Floating Glass Widget */
.pose-card {
    background: rgba(167, 139, 250, 0.05);
    border: 1px solid rgba(167, 139, 250, 0.15);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px -4px rgba(0,0,0,0.5);
    transition: all 0.3s ease;
}
.pose-card:hover {
    border-color: rgba(167, 139, 250, 0.3);
    background: rgba(167, 139, 250, 0.08);
}
.pose-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6, #d946ef, transparent);
}
.pose-name {
    font-family: 'Outfit', sans-serif;
    font-size: 26px;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.2;
    text-transform: capitalize;
    margin-bottom: 12px;
}
.pose-name.idle {
    font-size: 16px;
    color: rgba(255,255,255,0.4);
    font-weight: 400;
    font-family: 'Inter', sans-serif;
}
.confidence-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
}
.confidence-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #8b5cf6, #d946ef);
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px rgba(217, 70, 239, 0.3);
}
.confidence-pct {
    font-size: 14px;
    font-weight: 600;
    color: #e879f9;
    min-width: 42px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
}
.status-pill.active {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.1);
}
.status-pill.idle {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: rgba(255,255,255,0.5);
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
}
.status-dot.pulse {
    animation: smoothPulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
@keyframes smoothPulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* Feedback block - Glass */
.feedback-block {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 16px 20px;
}
.feedback-item {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    font-size: 15px;
    color: rgba(255,255,255,0.9);
    line-height: 1.5;
    font-weight: 400;
}
.feedback-item:last-child { border-bottom: none; }
.feedback-icon { color: #f59e0b; flex-shrink: 0; margin-top: 2px; font-size: 16px; }
.feedback-good {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 15px;
    color: #34d399;
    padding: 6px 0;
    font-weight: 500;
}

/* Streamlit Override */
[data-testid="stCheckbox"] {
    background: rgba(139, 92, 246, 0.1) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stCheckbox"]:hover {
    border-color: rgba(139, 92, 246, 0.4) !important;
    background: rgba(139, 92, 246, 0.15) !important;
}
[data-testid="stCheckbox"] label {
    color: #ffffff !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}
[data-testid="stCheckbox"] span[data-testid="stCheckboxValue"] {
    background: #8b5cf6 !important;
}

[data-testid="stSlider"] > div > div > div { color: #8b5cf6 !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #8b5cf6 !important; }

/* Stats row */
.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}
.stat-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    transition: all 0.3s ease;
}
.stat-box:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.15);
}
.stat-box-val {
    font-family: 'Outfit', sans-serif;
    font-size: 28px;
    font-weight: 600;
    color: #ffffff;
    line-height: 1;
}
.stat-box-label {
    font-size: 11px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5);
    margin-top: 6px;
    font-weight: 600;
}

/* Streamlit image override */
[data-testid="stImage"] {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stImage"] img {
    border-radius: 20px;
}

/* Layout container fix */
[data-testid="column"] { padding: 0 16px; }
</style>
""", unsafe_allow_html=True)

# Removed global model loading to avoid thread-safety issues, but classes can be loaded safely
classes = np.load('models/label_classes.npy', allow_pickle=True)
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def compute_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))

def extract_features(lm):
    def pt(i): return [lm[i].x, lm[i].y]
    raw = []
    for p in lm:
        raw.extend([p.x, p.y, p.z, p.visibility])
    angles = [
        compute_angle(pt(11), pt(13), pt(15)),
        compute_angle(pt(12), pt(14), pt(16)),
        compute_angle(pt(13), pt(11), pt(23)),
        compute_angle(pt(14), pt(12), pt(24)),
        compute_angle(pt(11), pt(23), pt(25)),
        compute_angle(pt(12), pt(24), pt(26)),
        compute_angle(pt(23), pt(25), pt(27)),
        compute_angle(pt(24), pt(26), pt(28)),
        compute_angle(pt(25), pt(27), pt(29)),
        compute_angle(pt(26), pt(28), pt(30)),
        compute_angle(pt(11), pt(12), pt(24)),
        compute_angle(pt(12), pt(11), pt(23)),
        compute_angle(pt(23), pt(24), pt(26)),
        compute_angle(pt(24), pt(23), pt(25)),
        compute_angle(pt(11), pt(23), pt(24)),
    ]
    return np.array(raw + angles, dtype=np.float32), angles

IDEAL_ANGLES = {
    "virabhadrasana ii": [170,170,90,170,90,170,150,170,90,90,45,135,90,170,90],
    "tadasana":          [170,170,10,10,170,170,170,170,90,90,10,10,170,170,90],
    "vriksasana":        [170,170,10,10,170,170,170,10,90,90,10,10,170,170,90],
}
JOINT_NAMES = ["L elbow","R elbow","L shoulder","R shoulder","L hip","R hip",
               "L knee","R knee","L ankle","R ankle","shoulder-hip R",
               "shoulder-hip L","hip-knee R","hip-knee L","torso"]

def get_feedback(predicted_class, angles):
    key = predicted_class.lower().strip()
    if key not in IDEAL_ANGLES:
        return []
    ideal = IDEAL_ANGLES[key]
    tips = []
    for i, (actual, target) in enumerate(zip(angles, ideal)):
        diff = abs(actual - target)
        if diff > 15:
            direction = "more" if actual < target else "less"
            tips.append(f"Adjust {JOINT_NAMES[i]} — bend {direction} ({diff:.0f}° off)")
    return tips[:3]

# ── Session state ─────────────────────────────────────────────────────────
if 'frames_processed' not in st.session_state:
    st.session_state.frames_processed = 0
if 'detections' not in st.session_state:
    st.session_state.detections = 0

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-tag">Computer Vision · Real-time</div>
        <div class="hero-title">Yoga<em>Vision</em></div>
        <div class="hero-sub">Pose classification & corrective feedback</div>
    </div>
    <div class="hero-stat">
        <div class="hero-stat-num">{len(classes)}</div>
        <div class="hero-stat-label">Pose classes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────
col_video, col_side = st.columns([3, 1.1])

with col_side:
    st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)
    threshold = st.slider("Min confidence", 0.0, 1.0, 0.4, 0.05, label_visibility="visible")

    st.markdown('<div class="section-label" style="margin-top:8px">Status</div>', unsafe_allow_html=True)
    status_placeholder = st.empty()

    st.markdown('<div class="section-label" style="margin-top:8px">Detected pose</div>', unsafe_allow_html=True)
    pose_placeholder = st.empty()

    st.markdown('<div class="section-label" style="margin-top:8px">Form feedback</div>', unsafe_allow_html=True)
    feedback_placeholder = st.empty()

    st.markdown('<div class="section-label" style="margin-top:8px">Session</div>', unsafe_allow_html=True)
    stats_placeholder = st.empty()

# ── Initial states ────────────────────────────────────────────────────────
status_placeholder.markdown("""
<div class="status-pill idle">
    <div class="status-dot"></div> Camera off
</div>""", unsafe_allow_html=True)

pose_placeholder.markdown("""
<div class="pose-card">
    <div class="pose-name idle">Waiting for pose…</div>
</div>""", unsafe_allow_html=True)

feedback_placeholder.markdown("""
<div class="feedback-block">
    <div class="feedback-item">
        <span style="color:rgba(232,228,220,0.7);font-size:15px;font-style:italic;">
            Activate camera to begin
        </span>
    </div>
</div>""", unsafe_allow_html=True)

stats_placeholder.markdown(f"""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-box-val">0</div>
        <div class="stat-box-label">Frames</div>
    </div>
    <div class="stat-box">
        <div class="stat-box-val">0</div>
        <div class="stat-box-label">Detections</div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Main loop ─────────────────────────────────────────────────────────────
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if 'result_queue' not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# We only want to instantiate the pose model once per thread, so we'll do it globally
# or inside a processor class. Let's use a processor class.
class YogaProcessor:
    def __init__(self):
        self.pose = None
        self.model = None
        self.classes = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if self.model is None:
            self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
            self.custom_style = mp_draw.DrawingSpec(color=(139, 108, 247), thickness=2, circle_radius=3)
            self.conn_style = mp_draw.DrawingSpec(color=(80, 60, 160), thickness=1)
            self.model = tf.keras.models.load_model('models/yoga_model.keras')
            self.classes = np.load('models/label_classes.npy', allow_pickle=True)

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(img_rgb)

        label_text = ""
        conf_val = 0.0
        feedback = []

        if result.pose_landmarks:
            mp_draw.draw_landmarks(img, result.pose_landmarks,
                                   mp_pose.POSE_CONNECTIONS,
                                   self.custom_style, self.conn_style)
            feats, angles = extract_features(result.pose_landmarks.landmark)
            probs = self.model(feats[np.newaxis], training=False)[0].numpy()
            conf_val = float(probs.max())
            if conf_val >= threshold:
                pred = self.classes[probs.argmax()]
                label_text = pred
                feedback = get_feedback(pred, angles)

        if label_text:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img.shape[1], 56), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.putText(img, label_text.title(), (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 180, 255), 2)
            cv2.putText(img, f"{conf_val*100:.1f}%", (img.shape[1]-80, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (139, 108, 247), 2)

        st.session_state.result_queue.put((label_text, conf_val, feedback))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="yoga",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YogaProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if webrtc_ctx.state.playing:
    status_placeholder.markdown("""
    <div class="status-pill active">
        <div class="status-dot pulse"></div> Live
    </div>""", unsafe_allow_html=True)

    while webrtc_ctx.state.playing:
        try:
            label_text, conf_val, feedback = st.session_state.result_queue.get(timeout=0.1)
            
            st.session_state.frames_processed += 1
            if label_text:
                st.session_state.detections += 1
                bar_w = int(conf_val * 100)
                pose_placeholder.markdown(f"""
                <div class="pose-card">
                    <div class="pose-name">{label_text.title()}</div>
                    <div class="confidence-row">
                        <div class="confidence-bar-bg">
                            <div class="confidence-bar-fill" style="width:{bar_w}%"></div>
                        </div>
                        <div class="confidence-pct">{conf_val*100:.1f}%</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                pose_placeholder.markdown("""
                <div class="pose-card">
                    <div class="pose-name idle">No pose detected…</div>
                </div>""", unsafe_allow_html=True)

            if feedback:
                items = "".join([f'<div class="feedback-item"><span class="feedback-icon">⚠</span>{f}</div>' for f in feedback])
                feedback_placeholder.markdown(f'<div class="feedback-block">{items}</div>', unsafe_allow_html=True)
            elif label_text:
                feedback_placeholder.markdown("""
                <div class="feedback-block">
                    <div class="feedback-good">✓ Form looks good</div>
                </div>""", unsafe_allow_html=True)

            stats_placeholder.markdown(f"""
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-box-val">{st.session_state.frames_processed}</div>
                    <div class="stat-box-label">Frames</div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-val">{st.session_state.detections}</div>
                    <div class="stat-box-label">Detections</div>
                </div>
            </div>""", unsafe_allow_html=True)
            
        except queue.Empty:
            continue
else:
    status_placeholder.markdown("""
    <div class="status-pill idle">
        <div class="status-dot"></div> Camera off
    </div>""", unsafe_allow_html=True)