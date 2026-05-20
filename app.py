"""
Aura — Vision Accessibility Core (Final Viva Edition)
Tier 1: Standard YOLO11m (COCO 80-Class)
Tier 2: Gemini 1.5 Flash (Latest Endpoint)
"""

import cv2
import os
import time
import base64
import queue
import threading

from flask import Flask, render_template
from flask_socketio import SocketIO
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
# 🛑 PASTE YOUR API KEY HERE 🛑
genai.configure(api_key="")  

app = Flask(__name__)
app.config["SECRET_KEY"] = "aura_viva_secret"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ─── Global State ─────────────────────────────────────────────────────────────

print("[EyeSpeak] Loading Standard YOLO11m (80 COCO Classes)...")
model = YOLO("yolo11m.pt")

# Standard COCO classes that will trigger an audio warning if they get too close.
HAZARDS = ['person', 'bottle','laptop', 'keyboard', 'cell phone', 'chair', 'couch', 'bed', 'book','stop sign','backpack']

latest_frame = None            
frame_lock = threading.Lock()  
scan_lock = threading.Lock()   
audio_queue = queue.Queue()    
is_scanning = False            

# ─── Audio TTS Worker ─────────────────────────────────────────────────────────
def audio_worker():
    while True:
        text = audio_queue.get()
        if text is None: break
        safe_text = text.replace("'", "'\\''")
        os.system(f"say '{safe_text}'")
        audio_queue.task_done()

# ─── Tier 1: Live YOLO Camera Stream ──────────────────────────────────────────
def camera_stream():
    global latest_frame, is_scanning

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()
    spoken_objects = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (640, 480))

        with frame_lock:
            latest_frame = frame.copy()

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 0.0001)
        prev_time = current_time

        with scan_lock:
            currently_scanning = is_scanning

        if currently_scanning:
            # Bypass YOLO to save CPU for Gemini
            ret_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret_enc:
                b64 = base64.b64encode(buffer).decode("utf-8")
                socketio.emit("video_frame", {"image": b64})
                socketio.emit("stats_update", {"fps": round(fps, 1), "objects": [], "paused": True})
        else:
            try:
                # Lowered confidence to 0.30 so it easily detects items in room lighting
                results = model.predict(frame, conf=0.30, imgsz=640, device="mps", verbose=False)
                annotated = results[0].plot()

                objects = []
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]

                    bx = float(box.xywh[0][0])
                    bw = float(box.xywh[0][2])
                    bh = float(box.xywh[0][3])
                    area_ratio = (bw * bh) / (640 * 480)
                    is_close = area_ratio > 0.3

                    if bx < 213: position = "left"
                    elif bx > 426: position = "right"
                    else: position = "center"

                    objects.append({
                        "label": label,
                        "confidence": round(conf * 100, 1),
                        "close": is_close,
                        "position": position,
                    })

                    # FIX: Explicit Audio String Formatting
                    if label in HAZARDS and is_close:
                        if label not in spoken_objects or (current_time - spoken_objects[label] > 5):
                            # This guarantees it says the actual name of the object
                            audio_queue.put(f"Caution. {label} directly ahead.")
                            spoken_objects[label] = current_time

                ret_enc, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret_enc:
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    socketio.emit("video_frame", {"image": b64})
                    socketio.emit("stats_update", {"fps": round(fps, 1), "objects": objects, "paused": False})

            except Exception as e:
                ret_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ret_enc:
                    b64 = base64.b64encode(buffer).decode("utf-8")
                    socketio.emit("video_frame", {"image": b64})

        time.sleep(0.033)
    cap.release()

# ─── Tier 2: Gemini VLM ───────────────────────────────────────────────────────
def run_vlm_describe():
    global is_scanning

    with scan_lock:
        is_scanning = True
    socketio.emit("vlm_status", {"status": "Scanning", "text": "Capturing frame and sending to Gemini…"})

    try:
        time.sleep(0.3)
        with frame_lock:
            if latest_frame is None: raise RuntimeError("No camera frame available.")
            frame_to_send = latest_frame.copy()

        rgb_frame = cv2.cvtColor(frame_to_send, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        NAVIGATION_PROMPT = (
            "You are a highly tactical navigation assistant for a visually impaired user. "
            "Analyze this image and provide a 2-sentence spatial breakdown. "
            "Rule 1: Ignore colors, lighting, and visual aesthetics entirely. "
            "Rule 2: In Sentence 1, declare any immediate obstacles directly in the center path. "
            "Rule 3: In Sentence 2, declare the surrounding context using strictly "
            "'on the left' or 'on the right'. "
            "Keep it brief, sterile, and focused purely on physical navigation."
        )

        # FIX: Appended '-latest' to resolve the 404 API Error
        vlm_model = genai.GenerativeModel("gemini-2.5-flash")
        response = vlm_model.generate_content([NAVIGATION_PROMPT, pil_image])

        description = response.text.strip()
        socketio.emit("vlm_status", {"status": "Done", "text": description})
        audio_queue.put(description)

    except Exception as e:
        error_msg = f"VLM Error: {str(e)}"
        print(f"[EyeSpeak] {error_msg}")
        socketio.emit("vlm_status", {"status": "Error", "text": error_msg})
        audio_queue.put("Network error. Scene analysis failed.")

    finally:
        with scan_lock:
            is_scanning = False

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("trigger_scan")
def handle_trigger_scan():
    global is_scanning
    if not is_scanning:
        threading.Thread(target=run_vlm_describe, daemon=True).start()

if __name__ == "__main__":
    threading.Thread(target=audio_worker, daemon=True).start()
    threading.Thread(target=camera_stream, daemon=True).start()
    print("[EyeSpeak] Starting server on http://localhost:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)