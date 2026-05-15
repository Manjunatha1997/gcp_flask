from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import os

app = Flask(__name__)

# Load YOLO model once
model = YOLO("yolov8n.pt")  # replace with your custom weights if needed


def read_image(file):
    """Convert uploaded file to OpenCV image"""
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return img


def encode_image(image):
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image = read_image(file)

    # Run YOLO inference
    results = model(image)[0]

    detections = []

    # Parse detections
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detections.append({
            "class": label,
            "confidence": conf,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

    # 🔥 Draw predictions on image
    annotated_image = results.plot()  # Ultralytics built-in renderer

    # Encode image to base64
    image_base64 = encode_image(annotated_image)

    return jsonify({
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image": image_base64  # send image as string
    })



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )