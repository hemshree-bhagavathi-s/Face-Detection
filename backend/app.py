"""
==============================================================================
Face Detection System - Python Flask Backend API
==============================================================================
This Flask application serves as the prediction backend for the Face Detection
System. It loads a trained TensorFlow/Keras CNN model (face_detector.h5) once at
startup, receives Base64-encoded webcam frames from the frontend, uses OpenCV's
Haar Cascade classifier to detect faces, crops & preprocesses the face region,
and passes it to the CNN model to return detection status and confidence scores.

Render Deployment Ready | Production Compliant
==============================================================================
"""

import os
import base64
import logging
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.clear_session()

# Configure Logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# Initialize Flask App
app = Flask(__name__)

# Enable CORS for all routes (allows browser frontend requests)
CORS(app)

# ============================================================================
# Global Initialization (Load Models Once at Startup)
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_detector.h5')
CASCADE_PATH = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')

# 1. Load OpenCV Haar Cascade Classifier for Face Localization
if os.path.exists(CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    logging.info(f"✅ Loaded OpenCV Haar Cascade from: {CASCADE_PATH}")
else:
    # Fallback to OpenCV built-in cascade path if local file missing
    cascade_builtin = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_builtin)
    logging.info(f"ℹ️ Loaded OpenCV Haar Cascade from built-in path: {cascade_builtin}")

if face_cascade.empty():
    logging.error("❌ Failed to load OpenCV Haar Cascade classifier!")

# 2. Load Trained Keras CNN Model (face_detector.h5)
cnn_model = None
TARGET_SIZE = (128, 128)  # Default fallback input target size (Width, Height)

try:
    if os.path.exists(MODEL_PATH):
        cnn_model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"✅ CNN Model '{MODEL_PATH}' loaded successfully.")
        
        # Dynamically infer CNN input shape from model configuration
        try:
            input_shape = cnn_model.input_shape
            # input_shape typically (None, H, W, C)
            if input_shape and len(input_shape) >= 3 and input_shape[1] is not None:
                h, w = input_shape[1], input_shape[2]
                TARGET_SIZE = (w, h)
                logging.info(f"📐 CNN Model Input Dimensions Detected: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
        except Exception as shape_err:
            logging.warning(f"Could not automatically parse input shape: {shape_err}. Using default 128x128.")
    else:
        logging.error(f"❌ Model file '{MODEL_PATH}' not found!")
except Exception as model_err:
    logging.error(f"❌ Error loading Keras CNN model: {str(model_err)}")

# ============================================================================
# Helper Functions
# ============================================================================
def decode_base64_image(base64_string):
    """
    Decodes a Base64 encoded image string into an OpenCV BGR numpy array.
    Supports both raw base64 strings and Data URL strings (e.g. data:image/jpeg;base64,...).
    """
    if not base64_string:
        raise ValueError("Image string is empty")

    # Remove Data URI scheme header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]

    # Clean whitespace and newlines
    base64_string = base64_string.strip()

    # Base64 decode to bytes
    image_bytes = base64.b64decode(base64_string)

    # Convert bytes to numpy uint8 array
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode image using OpenCV (BGR format)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Decoded image is invalid or corrupted")

    return img


def preprocess_face_crop(face_bgr, target_size=TARGET_SIZE):
    """
    Preprocesses cropped face BGR image for CNN model input:
    1. Converts BGR to RGB
    2. Resizes to model target resolution (e.g. 128x128)
    3. Normalizes pixel values to [0.0, 1.0] range
    4. Expands dimensions to create batch tensor of shape (1, H, W, 3)
    """
    # Convert BGR to RGB (Keras models usually expect RGB input)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Resize image to CNN input dimensions
    face_resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)

    # Convert to float32 and normalize to range [0.0, 1.0]
    face_normalized = face_resized.astype(np.float32) / 255.0

    # Expand dims to add batch dimension (1, height, width, 3)
    tensor_input = np.expand_dims(face_normalized, axis=0)

    return tensor_input


# ============================================================================
# API Routes
# ============================================================================
@app.route('/', methods=['GET'])
def index():
    """Health check route to verify backend service status."""
    return jsonify({
        "status": "online",
        "service": "Face Detection System CNN API",
        "model_loaded": cnn_model is not None,
        "cascade_loaded": not face_cascade.empty(),
        "input_resolution": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Primary API Endpoint: POST /predict
    Receives JSON body: { "image": "base64_encoded_string" }
    Returns JSON body:
      - If no face found: { "face_detected": false, "confidence": 0 }
      - If face found:    { "face_detected": true,  "confidence": 96.4 }
    """
    # 1. Check if model is available
    if cnn_model is None:
        return jsonify({
            "error": "CNN Model failed to load on server. Check server logs.",
            "face_detected": False,
            "confidence": 0
        }), 500

    # 2. Validate JSON Request Body
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data or not data['image']:
            return jsonify({
                "error": "Invalid request payload. 'image' field (base64 string) is required.",
                "face_detected": False,
                "confidence": 0
            }), 400

        base64_image = data['image']

    except Exception as parse_err:
        return jsonify({
            "error": f"Failed to parse JSON request: {str(parse_err)}",
            "face_detected": False,
            "confidence": 0
        }), 400

    # 3. Base64 Decode Image
    try:
        img = decode_base64_image(base64_image)
    except Exception as img_err:
        return jsonify({
            "error": f"Invalid base64 image data: {str(img_err)}",
            "face_detected": False,
            "confidence": 0
        }), 400

    # 4. Face Detection via OpenCV Haar Cascade
    try:
        # Convert BGR image to Grayscale for Haar Cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram to improve detection under varying lighting
        gray_eq = cv2.equalizeHist(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(50, 50)
            )

        # If Haar Cascade detects NO faces, return face_detected: false
        global face_history
        if len(faces) == 0:
            face_history.append(0)

            if len(face_history) > 10:
                face_history.pop(0)

                return jsonify({
                    "face_detected": sum(face_history) >= 5,
                    "confidence": 0
                    }), 200

        # Select the largest face region if multiple faces detected
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Add small padding around cropped face region for better feature context
        img_h, img_w = img.shape[:2]
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)

        # Crop face region
        cropped_face = img[y1:y2, x1:x2]

        if cropped_face.size == 0:
            return jsonify({
                "face_detected": False,
                "confidence": 0
            }), 200

        # 5. Preprocess Face Crop & Run CNN Model Prediction
        input_tensor = preprocess_face_crop(cropped_face, TARGET_SIZE)
        
        # Predict using trained CNN model
        raw_prediction = cnn_model.predict(input_tensor, verbose=0)
        
        # Extract prediction probability
        # Handles binary output shape (1, 1) or softmax categorical shape (1, 2)
        if raw_prediction.shape[-1] == 1:
            raw_score = float(raw_prediction[0][0])
            is_face = raw_score >= 0.85
            
            confidence_val = raw_score if is_face else (1.0 - raw_score)
        else:
            # Categorical prediction (e.g. index 1 = face)
            probs = raw_prediction[0]
            face_class_idx = np.argmax(probs)
            raw_score = float(probs[face_class_idx])
            is_face = (face_class_idx == 1) or (raw_score > 0.5)
            confidence_val = raw_score

        # Format confidence score percentage rounded to 1 decimal place (e.g. 96.4)
        confidence_percentage = round(float(confidence_val * 100), 1)
        face_history.append(1)

        if len(face_history) > 10:
            face_history.pop(0)

            return jsonify({
                "face_detected": sum(face_history) >= 5,
                "confidence": confidence_percentage
                }), 200

    except Exception as proc_err:
        logging.error(f"Prediction Processing Error: {str(proc_err)}", exc_info=True)
        return jsonify({
            "error": f"Internal image processing error: {str(proc_err)}",
            "face_detected": False,
            "confidence": 0
        }), 500


# ============================================================================
# Custom Error Handlers
# ============================================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Route not found",
        "message": "Valid endpoints are GET / and POST /predict"
    }), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({
        "error": "Internal Server Error",
        "message": str(error)
    }), 500


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == '__main__':
    # Retrieve PORT from environment variables (Render sets PORT dynamically)
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print("\n" + "=" * 60)
    print("🚀 Face Detection System - Python Flask Backend Server")
    print(f"📍 Running locally on: http://127.0.0.1:{port}")
    print(f"⚡ Endpoint: POST http://127.0.0.1:{port}/predict")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
