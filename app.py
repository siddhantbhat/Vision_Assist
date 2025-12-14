from flask import Flask, render_template, Response, jsonify
import cv2
from yolo_detector import detect_objects
from gemini_explainer import explain
from datetime import datetime

app = Flask(__name__)

camera = cv2.VideoCapture(0)
latest_objects = []

def generate_frames():
    global latest_objects

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Detect objects
        latest_objects = detect_objects(frame)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/system-message')
def system_message():
    if latest_objects:
        return jsonify({
            "status": "success",
            "message": f"I can see: {', '.join(latest_objects)}",
            "timestamp": datetime.now().isoformat()
        })
    else:
        return jsonify({
            "status": "success",
            "message": "No objects detected",
            "timestamp": datetime.now().isoformat()
        })

if __name__ == "__main__":
    app.run(debug=True)
