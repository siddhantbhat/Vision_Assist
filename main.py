from flask import Blueprint, render_template, Response, jsonify
import cv2
import threading
import queue
import time
import pyttsx3
import speech_recognition as sr

# ===== SAFE IMPORTS =====
try:
    from yolo_detector import detect_objects
    from gemini_explainer import explain
except ImportError as e:
    print(f"Warning: {e}")
    def detect_objects(frame):
        return [], frame
    def explain(objects):
        return "Module not found"

main_bp = Blueprint("main", __name__)

# ===== GLOBAL STATE =====
camera = None
camera_on = False
latest_objects = []
spoken_objects = set()

# ===== TTS SETUP (WINDOWS SAFE) =====
speech_queue = queue.Queue()

engine = pyttsx3.init(driverName="sapi5")
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

def tts_worker():
    while True:
        text = speech_queue.get()
        if text:
            engine.say(text)
            engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    if speech_queue.empty():
        speech_queue.put(text)

# ===== ROUTES =====

@main_bp.route("/")
def home():
    return render_template("index.html")

@main_bp.route("/camera/start")
def start_camera():
    global camera, camera_on, spoken_objects
    if not camera_on:
        camera = cv2.VideoCapture(0)
        camera_on = True
        spoken_objects.clear()
    return jsonify({"status": "camera started"})

@main_bp.route("/camera/stop")
def stop_camera():
    global camera, camera_on, spoken_objects
    if camera_on and camera:
        camera.release()
    camera_on = False
    spoken_objects.clear()
    return jsonify({"status": "camera stopped"})

# ===== VIDEO STREAM =====

def generate_frames():
    global latest_objects, spoken_objects

    while True:
        if not camera_on or camera is None:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            continue

        # YOLO detection (labels + annotated frame)
        latest_objects, frame = detect_objects(frame)

        # 🔊 SPEAK NEW OBJECTS
        for obj in latest_objects:
            if obj not in spoken_objects:
                speak(f"I can see {obj}")
                spoken_objects.add(obj)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

@main_bp.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ===== GEMINI EXPLANATION =====

@main_bp.route("/api/explain")
def explain_scene():
    return jsonify({
        "explanation": explain(latest_objects)
    })

# ===== VOICE COMMANDS =====

def voice_listener():
    r = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source)

            text = r.recognize_google(audio).lower()
            print("Voice command:", text)

            if "start camera" in text:
                start_camera()
            elif "stop camera" in text:
                stop_camera()

        except Exception:
            pass

threading.Thread(target=voice_listener, daemon=True).start()
# Note: Ensure that the necessary modules (yolo_detector, gemini_explainer) are available in your environment.