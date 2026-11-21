import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import base64
import numpy as np
import time
import csv
import zipfile
from io import BytesIO, StringIO
from datetime import datetime
import yt_dlp

st.set_page_config(page_title="Flood Level Detection", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #e3f2fd 0%, #ffffff 25%, #ffffff 75%, #e3f2fd 100%);
        }
        .stApp {
            background: linear-gradient(to right, #e3f2fd 0%, #ffffff 25%, #ffffff 75%, #e3f2fd 100%);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 8rem;}
        .main-title {font-size:34px;font-weight:bold;text-align:center;color:#004080;margin-top:40px;margin-bottom:35px;}
        .settings-box {border:2px solid #363738;border-radius:10px;padding:15px;background-color:#E8E8E8;}
        .yellow-header {background-color:#363738;color:white;font-weight:bold;text-align:center;padding:6px;border-radius:6px;margin-bottom:10px;}
        .centered-status {display:flex;justify-content:center;align-items:center;margin-top:15px;}
        .status-text {text-align:center;font-size:16px;font-weight:bold;}
        .progress-container {display:flex;justify-content:center;align-items:center;width:100%;margin-top:15px;}
        .progress-bar-wrapper {width:60%;}
        div[data-testid="stButton"] > button {
            background: linear-gradient(to right, #e3f2fd 0%, #ffffff 25%, #ffffff 75%, #e3f2fd 100%) !important;
            color: #004080 !important;
            font-weight: bold !important;
            border: 2px solid #004080 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(to right, #bbdefb 0%, #e3f2fd 25%, #e3f2fd 75%, #bbdefb 100%) !important;
            border-color: #0d47a1 !important;
            box-shadow: 0 4px 12px rgba(13, 71, 161, 0.2) !important;
        }
        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(to right, #e3f2fd 0%, #ffffff 25%, #ffffff 75%, #e3f2fd 100%) !important;
            color: #004080 !important;
            font-weight: bold !important;
            border: 2px solid #004080 !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            background: linear-gradient(to right, #bbdefb 0%, #e3f2fd 25%, #e3f2fd 75%, #bbdefb 100%) !important;
            border-color: #0d47a1 !important;
            box-shadow: 0 4px 12px rgba(13, 71, 161, 0.2) !important;
        }
        div[data-testid="stNumberInput"] label {
            font-size: 20px !important;
            font-weight: bold !important;
            color: #004080 !important;
        }
        div[data-testid="stNumberInput"] input {
            font-size: 24px !important;
            font-weight: bold !important;
            height: 50px !important;
        }
        div[data-testid="stRadio"] label {
            font-size: 18px !important;
            font-weight: bold !important;
            color: #004080 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .header-container {
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            margin-top: 30px;
            margin-bottom: 25px;
            background: transparent;
            padding: 0px 0;
            border: none;
        }
        .header-title {
            font-size: 36px;
            font-weight: bold;
            color: #004080;
            text-align: center;
        }
        .header-logo {
            position: absolute;
            right: 80px;
            top: 50%;
            transform: translateY(-50%);
        }
        .header-logo img {
            height: 70px;
            width: auto;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# # --- Load logo ---
# with open("assets/logo3u.png", "rb") as img_file:
#     logo_data = base64.b64encode(img_file.read()).decode()

# # --- Header layout ---
# st.markdown(f"""
#     <style>
#         .header-logo img {{
#             height: 120px;  /* increase this value to make the image bigger */
#         }}
#     </style>

#     <div class="header-container">
#         <div class="header-title">FLOOD-DEPTH-ML</div>
#         <div class="header-logo">
#             <img src="data:image/png;base64,{logo_data}">
#         </div>
#     </div>
# """, unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL..Here you change the bestweights.file ....
# ---------------------------
MODEL_PATH = "best_car.pt"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found! Please place your model as 'best_car.pt'")
    st.stop()
model = YOLO(MODEL_PATH)

# ---------------------------
# LAYOUT
# ---------------------------
col1, col2, col3 = st.columns([1.2, 2.6, 1.2])

# ---------------------------
# LEFT PANEL
# ---------------------------
with col1:
    st.subheader("📥 Input Source")
    input_type = st.radio("Choose Input Type", ["Upload File", "YouTube Link", "Use Webcam"])
    uploaded_file = None
    youtube_url = None

    if input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4", "avi"])
        analyze_btn = st.button("🔍 Analyze", width='stretch')
    elif input_type == "YouTube Link":
        youtube_url = st.text_input("🔗 Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        analyze_btn = st.button("🔍 Analyze YouTube Video", width='stretch')
    else:
        # Webcam controls - compact layout
        st.markdown("##### 📷 Live Webcam Controls")
        colA, colB, colC = st.columns(3)
        with colA:
            start_webcam = st.button("▶️ Start", help="Start/Resume", use_container_width=True)
        with colB:
            stop_webcam = st.button("⏸️ Pause", help="Pause", use_container_width=True)
        with colC:
            exit_webcam = st.button("⏹ Stop", help="Stop", use_container_width=True)

        # Recording option
        record_webcam = st.checkbox("💾 Record Session", key="record_webcam", help="Save webcam session as video + CSV report")

        analyze_btn = False  # Not used for webcam

    download_area = st.empty()

# ---------------------------
# CENTER PANEL
# ---------------------------
with col2:
    st.subheader("🎥 Detection Display")
    display_area = st.empty()
    controls_area = st.container()
    status_area = st.empty()

# ---------------------------
# RIGHT PANEL (settings)
# ---------------------------
with col3:
    st.markdown("<div class='settings-box'>", unsafe_allow_html=True)
    st.markdown("<div class='yellow-header'>⚙️ SETTINGS</div>", unsafe_allow_html=True)

    default_skip = 1
    try:
        raw_input = st.number_input(
            "⏱️ Analyze every Nth frame (1 = all, 10 = skip 10 frames)",
            min_value=1,
            max_value=60,
            value=default_skip,
            step=1,
            key="frame_skip"
        )
        fps_input = int(raw_input)
    except Exception:
        fps_input = default_skip

    st.markdown("### 🎯 Confidence Threshold")
    conf_threshold = st.slider(
        "Detection Confidence (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        key="conf_threshold",
        help="Minimum confidence score for detections (model's probability that object exists and is correctly classified)"
    )

    st.markdown("### 🌊 Flood Levels")
    level_placeholders = {f"Level {i}": st.empty() for i in range(5)}
    for level in level_placeholders:
        level_placeholders[level].markdown(
            f"<div style='text-align:left;font-size:20px;font-weight:bold;margin:4px;'>{level}: 0</div>",
            unsafe_allow_html=True
        )

    st.markdown("### 🧾 Labelling Criteria")
    if os.path.exists("assets/scheme.png"):
        st.image("assets/scheme.png", caption="Reference Criteria", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# FOOTER
# ---------------------------
# def show_footer_logos():
#     logo1 = "assets/logo1.png"
#     if not (os.path.exists(logo1)):
#         return
#     with open(logo1, "rb") as f:
#         a = base64.b64encode(f.read()).decode()
#     st.markdown(f"""
#         <div style="position:fixed;left:0;bottom:0;width:100%;background-color:white;display:flex;justify-content:center;align-items:center;gap:10px;padding:5px 0;box-shadow:0 -2px 8px rgba(0,0,0,0.15);z-index:999;">
#             <img src="data:image/png;base64,{a}" style="height:70px;">
#         </div>
#     """, unsafe_allow_html=True)


# show_footer_logos()


# ---------------------------
# HELPERS
# ---------------------------
def update_levels(counts, high_danger=False):
    for i, key in enumerate(level_placeholders.keys()):
        color = "#FF0000" if high_danger and key in ["Level 3", "Level 4"] else "#004080"
        level_placeholders[key].markdown(
            f"<div style='text-align:left;font-size:18px;font-weight:bold;margin:4px;color:{color};'>{key}: {counts.get(key, 0)}</div>",
            unsafe_allow_html=True
        )

if "pause_loop_counter" not in st.session_state:
    st.session_state.pause_loop_counter = 0
if "video_frame_position" not in st.session_state:
    st.session_state.video_frame_position = 0

def download_youtube_video(url, progress_bar=None, status_text=None):
    """Download YouTube video and return the path to the downloaded file."""
    try:
        # Create temp directory for download
        temp_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(temp_dir, f"youtube_video_{timestamp}.mp4")

        # Progress hook for yt-dlp
        def progress_hook(d):
            if progress_bar is not None and status_text is not None:
                if d['status'] == 'downloading':
                    # Extract percentage
                    percent_str = d.get('_percent_str', '0%').strip()
                    try:
                        percent = float(percent_str.replace('%', ''))
                        progress_bar.progress(percent / 100.0)

                        # Show download stats
                        downloaded = d.get('_downloaded_bytes_str', 'N/A')
                        total = d.get('_total_bytes_str', 'N/A')
                        speed = d.get('_speed_str', 'N/A')
                        eta = d.get('_eta_str', 'N/A')

                        status_text.text(f"📥 Downloading: {percent:.1f}% | {downloaded}/{total} | Speed: {speed} | ETA: {eta}")
                    except:
                        pass
                elif d['status'] == 'finished':
                    if progress_bar is not None:
                        progress_bar.progress(1.0)
                    if status_text is not None:
                        status_text.text("✅ Download complete! Processing video...")

        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'progress_hooks': [progress_hook],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return output_path, None
    except Exception as e:
        return None, str(e)

def detect_and_count(frame, conf_threshold=0.25):
    if frame is None:
        return None, {f"Level {i}": 0 for i in range(5)}, False

    # Run model with confidence threshold
    results = model(frame, conf=conf_threshold)

    # Manually draw annotations with confidence values
    annotated = frame.copy()
    level_counts = {f"Level {i}": 0 for i in range(5)}
    high_danger = False

    try:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()

            key = f"Level {cls}"
            if key in level_counts:
                level_counts[key] += 1
            if cls in [3, 4]:
                high_danger = True

            # Draw bounding box
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 0, 255) if cls in [3, 4] else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label with confidence
            label = f"Level {cls}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)

            # Background rectangle for text
            cv2.rectangle(annotated,
                         (x1, label_y - label_size[1] - 5),
                         (x1 + label_size[0], label_y + 5),
                         color, -1)

            # Put text
            cv2.putText(annotated, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    except Exception as e:
        print(f"Detection error: {e}")

    return annotated, level_counts, high_danger


# ---------------------------
# SESSION STATE INIT
# ---------------------------
if "processing" not in st.session_state:
    st.session_state.processing = False
if "tmp_video_path" not in st.session_state:
    st.session_state.tmp_video_path = None
if "paused" not in st.session_state:
    st.session_state.paused = False
if "zip_ready" not in st.session_state:
    st.session_state.zip_ready = False
if "zip_data" not in st.session_state:
    st.session_state.zip_data = None
if "zip_filename" not in st.session_state:
    st.session_state.zip_filename = None
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False
if "webcam_paused" not in st.session_state:
    st.session_state.webcam_paused = False
if "last_webcam_frame" not in st.session_state:
    st.session_state.last_webcam_frame = None
if "last_webcam_counts" not in st.session_state:
    st.session_state.last_webcam_counts = {}
if "report_log" not in st.session_state:
    st.session_state.report_log = []
if "webcam_frame_counter" not in st.session_state:
    st.session_state.webcam_frame_counter = 0
if "last_frame_bytes" not in st.session_state:
    st.session_state.last_frame_bytes = None
if "webcam_video_writer" not in st.session_state:
    st.session_state.webcam_video_writer = None
if "webcam_output_path" not in st.session_state:
    st.session_state.webcam_output_path = None
if "webcam_report_log" not in st.session_state:
    st.session_state.webcam_report_log = []
if "webcam_session_frame_count" not in st.session_state:
    st.session_state.webcam_session_frame_count = 0
if "webcam_zip_ready" not in st.session_state:
    st.session_state.webcam_zip_ready = False
if "webcam_zip_data" not in st.session_state:
    st.session_state.webcam_zip_data = None

# ---------------------------
# YOUTUBE & UPLOAD FILE PROCESSING
# ---------------------------
# Clean up webcam if switching to upload/youtube mode
if input_type in ["Upload File", "YouTube Link"]:
    st.session_state.webcam_running = False
    st.session_state.webcam_paused = False

# Handle YouTube video download and processing
if input_type == "YouTube Link" and analyze_btn and youtube_url:
    if not youtube_url.strip():
        st.error("❌ Please enter a valid YouTube URL")
    else:
        # Create progress bar and status text in the center display area
        with display_area.container():
            st.markdown("### 📥 Downloading YouTube Video")
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            status_text.text("🔄 Initializing download...")

        # Download with progress tracking
        video_path, error = download_youtube_video(youtube_url, progress_bar, status_text)

        if error:
            display_area.empty()
            st.error(f"❌ Failed to download YouTube video: {error}")
        elif video_path and os.path.exists(video_path):
            st.session_state.tmp_video_path = video_path
            st.session_state.processing = True
            st.session_state.paused = False
            st.session_state.zip_ready = False
            st.session_state.report_log = []
            st.session_state.video_frame_position = 0
            display_area.empty()
            status_area.success("✅ YouTube video downloaded successfully! Starting analysis...")
        else:
            display_area.empty()
            st.error("❌ Failed to download YouTube video")

if input_type == "Upload File" and analyze_btn and uploaded_file:
    ext = uploaded_file.name.split('.')[-1].lower()

    # Save file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix="." + ext)
    tfile.write(uploaded_file.read())
    tfile.close()
    st.session_state.tmp_video_path = tfile.name
    st.session_state.processing = True
    st.session_state.paused = False
    st.session_state.zip_ready = False
    st.session_state.report_log = []
    st.session_state.video_frame_position = 0

# Process uploaded file
if st.session_state.processing and st.session_state.tmp_video_path:
    path = st.session_state.tmp_video_path
    ext = path.split('.')[-1].lower()

    # IMAGE PROCESSING
    if ext in ["jpg", "jpeg", "png"]:
        frame = cv2.imread(path)
        if frame is None:
            st.error("Could not read image file")
        else:
            conf_thresh = st.session_state.get("conf_threshold", 50) / 100.0
            annotated, counts, high_danger = detect_and_count(frame, conf_thresh)
            update_levels(counts, high_danger)
            display_area.image(annotated, channels="BGR", caption="Detection Result", width='stretch')

            if high_danger:
                status_area.error("🚨 HIGH DANGER DETECTED!")

            _, enc = cv2.imencode('.jpg', annotated)
            with download_area:
                st.download_button("⬇️ Download Detected Image", enc.tobytes(),
                                   "detected_image.jpg", "image/jpeg", width='stretch')
            st.session_state.processing = False

    # VIDEO PROCESSING
    elif ext in ["mp4", "avi", "mov"]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            st.error("Could not open video file")
            st.session_state.processing = False
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = max(int(cap.get(cv2.CAP_PROP_FPS)), 20)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            frame_skip = st.session_state.get("frame_skip", 1)
            # Keep original FPS to maintain video duration
            output_fps = fps

            output_path = os.path.join(tempfile.gettempdir(), f"flood_output_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            if not video_writer.isOpened():
                st.error("❌ VideoWriter failed to open — check codec or path.")
                st.stop()

            # Restore frame position if resuming from pause
            frame_idx = st.session_state.video_frame_position
            processed_count = frame_idx // frame_skip
            total_to_process = max(1, total_frames // frame_skip)
            last_frame_bytes = None
            last_counts, last_danger = {}, False

            # Set video to correct position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            # Create controls and placeholders BEFORE the loop
            with controls_area:
                colA, colB, colC = st.columns(3)
                with colA:
                    pause_btn = st.button("⏸️ Pause", key="pause_btn", use_container_width=True)
                with colB:
                    resume_btn = st.button("▶️ Resume", key="resume_btn", use_container_width=True)
                with colC:
                    download_placeholder = st.empty()

            if pause_btn:
                st.session_state.paused = True
                st.session_state.pause_loop_counter = 0
            if resume_btn:
                st.session_state.paused = False
                st.session_state.pause_loop_counter = 0

            # Create centered progress bar
            prog_col1, prog_col2, prog_col3 = st.columns([0.5, 3, 0.5])
            with prog_col2:
                progress_bar = st.progress(0.0)

            # Initialize storage for last frame
            if 'last_frame_bytes' not in st.session_state:
                st.session_state.last_frame_bytes = None
                st.session_state.last_counts = {}
                st.session_state.last_danger = False

            # --- Main Loop ---
            while cap.isOpened() and frame_idx < total_frames:
                # Save current frame position to session state
                st.session_state.video_frame_position = frame_idx

                # Save last processed frame info to session state
                if last_frame_bytes:
                    st.session_state.last_frame_bytes = last_frame_bytes
                    st.session_state.last_counts = last_counts
                    st.session_state.last_danger = last_danger

                # --- Improved Pause Handling ---
                if st.session_state.paused:
                    if st.session_state.last_frame_bytes:
                        np_img = np.frombuffer(st.session_state.last_frame_bytes, np.uint8)
                        paused_frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                        display_area.image(paused_frame, channels="BGR", width='stretch')

                        update_levels(st.session_state.last_counts, st.session_state.last_danger)
                        status_area.info("⏸️ Video Paused ")

                        with download_placeholder.container():
                            st.download_button(
                                "⬇️ Download Paused Frame",
                                st.session_state.last_frame_bytes,
                                file_name=f"paused_frame_{frame_idx}.jpg",
                                mime="image/jpeg",
                                key=f"paused_dl_{st.session_state.pause_loop_counter}",
                                use_container_width=True
                            )
                        st.session_state.pause_loop_counter += 1
                    else:
                        status_area.info("No frame available yet.")
                    time.sleep(0.3)
                    continue
                else:
                    # Show download button during processing
                    if st.session_state.last_frame_bytes:
                        with download_placeholder.container():
                            st.download_button(
                                "⬇️ Download Current Frame",
                                st.session_state.last_frame_bytes,
                                file_name=f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg",
                                key=f"current_dl_{processed_count}",
                                use_container_width=True
                            )
                    else:
                        download_placeholder.empty()

                ret, frame = cap.read()
                if not ret:
                    break

                conf_thresh = st.session_state.get("conf_threshold", 50) / 100.0
                annotated, counts, high_danger = detect_and_count(frame, conf_thresh)
                if annotated is not None:
                    update_levels(counts, high_danger)
                    display_area.image(annotated, channels="BGR", width='stretch')

                    _, enc = cv2.imencode('.jpg', annotated)
                    last_frame_bytes = enc.tobytes()
                    last_counts, last_danger = counts, high_danger

                    # Write the processed frame multiple times to maintain original video duration
                    # If frame_skip=10, write this frame 10 times to fill the gap
                    for _ in range(frame_skip):
                        video_writer.write(annotated)

                    processed_count += 1
                    with prog_col2:
                        progress_bar.progress(min(processed_count / total_to_process, 1.0))

                    if high_danger:
                        status_area.markdown(f"<div class='centered-status'><div class='status-text' style='color:#FF0000;'>🚨 HIGH DANGER! Frame {processed_count}/{total_to_process}</div></div>", unsafe_allow_html=True)
                    else:
                        status_area.markdown(f"<div class='centered-status'><div class='status-text'>▶️ Processing frame {processed_count}/{total_to_process}</div></div>", unsafe_allow_html=True)

                    st.session_state.report_log.append([
                        frame_idx + 1,
                        counts.get("Level 0", 0),
                        counts.get("Level 1", 0),
                        counts.get("Level 2", 0),
                        counts.get("Level 3", 0),
                        counts.get("Level 4", 0)
                    ])

                frame_idx += frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                time.sleep(0.03)

            cap.release()
            video_writer.release()
            status_area.success("✅ Video processing finished!")
            st.session_state.processing = False

            # --- ZIP Export ---
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                if os.path.exists(output_path):
                    zipf.write(output_path, arcname="detected_video.mp4")

                csv_stream = StringIO()
                csv_writer = csv.writer(csv_stream)
                csv_writer.writerow(["Frame No", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4"])
                csv_writer.writerows(st.session_state.report_log)
                zipf.writestr("flood_level_report.csv", csv_stream.getvalue())

            zip_buffer.seek(0)
            st.session_state.zip_data = zip_buffer.getvalue()
            st.session_state.zip_filename = f"flood_analysis_{timestamp}.zip"
            st.session_state.zip_ready = True

            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                if os.path.exists(st.session_state.tmp_video_path):
                    os.remove(st.session_state.tmp_video_path)
            except Exception:
                pass

# ---------------------------
# WEBCAM PROCESSING
# ---------------------------
if input_type == "Use Webcam":
    # Button actions (buttons are defined in left panel)
    if start_webcam:
        st.session_state.webcam_running = True
        st.session_state.webcam_paused = False
        st.session_state.webcam_zip_ready = False

        # Initialize recording if checkbox is checked
        if record_webcam and st.session_state.webcam_video_writer is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.webcam_output_path = f"webcam_recording_{timestamp}.mp4"
            st.session_state.webcam_report_log = []
            st.session_state.webcam_session_frame_count = 0

    if stop_webcam:
        st.session_state.webcam_paused = True

    if exit_webcam:
        st.session_state.webcam_running = False
        st.session_state.webcam_paused = False

        # Finalize recording if active
        if st.session_state.webcam_video_writer is not None:
            st.session_state.webcam_video_writer.release()
            st.session_state.webcam_video_writer = None

            # Create ZIP with video and report
            if os.path.exists(st.session_state.webcam_output_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                    # Add video
                    zipf.write(st.session_state.webcam_output_path, arcname="webcam_recording.mp4")

                    # Add CSV report
                    report_stream = StringIO()
                    writer = csv.writer(report_stream)
                    writer.writerow(["Frame No", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4"])
                    for entry in st.session_state.webcam_report_log:
                        writer.writerow(entry)
                    zipf.writestr("webcam_report.csv", report_stream.getvalue().encode("utf-8"))

                zip_buffer.seek(0)
                st.session_state.webcam_zip_data = zip_buffer.read()
                st.session_state.webcam_zip_ready = True

                # Cleanup video file
                try:
                    os.remove(st.session_state.webcam_output_path)
                except:
                    pass

    # Main webcam loop
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_area.error("❌ Could not access webcam!")
            st.session_state.webcam_running = False
        else:
            # Initialize video writer for recording
            if record_webcam and st.session_state.webcam_video_writer is None:
                # Use fixed FPS for recording to maintain consistent playback
                # We'll control timing to match this FPS
                recording_fps = 20  # Fixed at 20 FPS for stable playback
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                st.session_state.webcam_video_writer = cv2.VideoWriter(
                    st.session_state.webcam_output_path,
                    fourcc,
                    recording_fps,
                    (frame_width, frame_height)
                )
                # Store recording FPS and timing info
                st.session_state.recording_fps = recording_fps
                st.session_state.frame_interval = 1.0 / recording_fps  # Time between frames
                st.session_state.last_frame_time = time.time()
                st.session_state.webcam_frame_counter_internal = 0
            while st.session_state.webcam_running:
                # Handle pause state
                if st.session_state.webcam_paused:
                    if st.session_state.last_webcam_frame is not None:
                        display_area.image(st.session_state.last_webcam_frame, channels="BGR", width='stretch')
                        update_levels(st.session_state.last_webcam_counts)
                        status_area.info("⏸️ Paused")
                    time.sleep(0.1)
                    continue

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    status_area.warning("⚠️ Could not read from webcam")
                    break

                # Get frame skip setting
                frame_skip = st.session_state.get("frame_skip", 1)

                # Increment internal frame counter
                if not hasattr(st.session_state, 'webcam_frame_counter_internal'):
                    st.session_state.webcam_frame_counter_internal = 0
                st.session_state.webcam_frame_counter_internal += 1

                # Check if we should process this frame for detection (based on frame skip)
                should_process_frame = (st.session_state.webcam_frame_counter_internal % frame_skip == 0)

                if should_process_frame:
                    # Detect with confidence threshold
                    conf_thresh = st.session_state.get("conf_threshold", 50) / 100.0
                    annotated, counts, high_danger = detect_and_count(frame, conf_thresh)
                    # Save for pause functionality
                    st.session_state.last_webcam_frame = annotated
                    st.session_state.last_webcam_counts = counts
                else:
                    # For skipped frames, use last detection results for display
                    annotated = frame
                    counts = st.session_state.get('last_webcam_counts', {f"Level {i}": 0 for i in range(5)})
                    high_danger = False

                # Save to video if recording - only write at correct time intervals to maintain real-time playback
                if record_webcam and st.session_state.webcam_video_writer is not None:
                    current_time = time.time()
                    time_since_last_frame = current_time - st.session_state.last_frame_time

                    # Only write frame if enough time has passed (based on recording FPS)
                    if time_since_last_frame >= st.session_state.frame_interval:
                        st.session_state.webcam_video_writer.write(annotated)
                        st.session_state.last_frame_time = current_time

                        # Only log to report if this frame was actually processed for detection
                        if should_process_frame:
                            st.session_state.webcam_session_frame_count += 1
                            st.session_state.webcam_report_log.append([
                                st.session_state.webcam_session_frame_count,
                                counts.get("Level 0", 0),
                                counts.get("Level 1", 0),
                                counts.get("Level 2", 0),
                                counts.get("Level 3", 0),
                                counts.get("Level 4", 0)
                            ])

                # Display
                display_area.image(annotated, channels="BGR", width='stretch')
                update_levels(counts, high_danger)

                if high_danger:
                    status_area.error("🚨 HIGH DANGER DETECTED!")
                else:
                    recording_status = " | 🔴 Recording" if record_webcam else ""
                    status_area.success(f"📹 Live Detection Active{recording_status}")

                # Download button for current frame
                _, enc = cv2.imencode('.jpg', annotated)
                with download_area:
                    st.download_button(
                        "📸 Capture Frame",
                        enc.tobytes(),
                        f"webcam_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                        "image/jpeg",
                        key=f"webcam_dl_{st.session_state.webcam_frame_counter}",
                        use_container_width=True
                    )
                st.session_state.webcam_frame_counter += 1

                # Smooth update
                time.sleep(0.03)

            cap.release()
    else:
        status_area.info("📹 Click ▶️ to start webcam")

# --- ZIP DOWNLOAD (left panel) ---
if st.session_state.zip_ready:
    with download_area:
        st.download_button(
            label="📦 Download ZIP (Video + Report)",
            data=st.session_state.zip_data,
            file_name=st.session_state.zip_filename,
            mime="application/zip",
            key="zip_download_button",
            use_container_width=True
        )

# --- WEBCAM ZIP DOWNLOAD (left panel) ---
if st.session_state.webcam_zip_ready:
    with download_area:
        st.download_button(
            label="📦 Download Webcam Session (Video + Report)",
            data=st.session_state.webcam_zip_data,
            file_name=f"webcam_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            key="webcam_zip_download_button",
            use_container_width=True
        )