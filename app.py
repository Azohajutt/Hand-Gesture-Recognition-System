from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
from cvzone.Utils import putTextRect

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("best.pt")

# Map original labels to display-friendly text
LABEL_MAP ={
    "Iloveyou": "I LOVE YOU",
    "hello": "HELLO",
    "no": "NO",
    "sleep": "SLEEP",
    "thankyou": "THANK YOU",
    "where": "WHERE",
    "yes": "YES"
}

def generate_frames():
    """Capture frames from webcam, perform detection, and yield encoded frames."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run inference on the current frame
        results = model(frame, imgsz=640, conf=0.3)[0]

        # Draw bounding boxes and labels
        for detection in results.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            confidence = float(detection.conf[0])
            class_id = int(detection.cls[0])
            class_name = model.names[class_id]
            display_label = LABEL_MAP.get(class_name, class_name)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
            putTextRect(
                frame,
                f"{display_label} ({confidence:.2f})",
                (x1, y1 - 10),
                scale=1,
                thickness=2,
                colorR=(255, 0, 255),
                colorT=(255, 255, 255)
            )

        # Encode and stream the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.route('/')
def index():
    """Render the homepage with camera feed."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    # In production, use a WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=5000, debug=True)
