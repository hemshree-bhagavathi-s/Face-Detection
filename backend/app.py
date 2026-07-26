"""
==============================================================================
Face Detection System - Flask Backend API
==============================================================================
"""

import os
import base64
import logging

import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS


# Disable GPU
tf.config.set_visible_devices([], "GPU")


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)


# Flask Application
app = Flask(__name__)

# Allow Firebase frontend access
CORS(app)

# ============================================================================
# Load Haar Cascade Face Detector
# ============================================================================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade model")


logging.info("Haar Cascade loaded successfully")



# ============================================================================
# Load TensorFlow Lite CNN Model
# ============================================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "face_detector.tflite"
)


TARGET_SIZE = (128, 128)



try:

    cnn_model = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    cnn_model.allocate_tensors()


    input_details = cnn_model.get_input_details()

    output_details = cnn_model.get_output_details()


    logging.info("CNN TFLite model loaded successfully")


except Exception as e:

    cnn_model = None

    logging.error(
        f"Model loading failed: {e}"
    )

    # ============================================================================
# Helper Functions
# ============================================================================


def decode_base64_image(base64_string):

    try:

        # Remove header if image comes as:
        # data:image/jpeg;base64,xxxx

        if "," in base64_string:
            base64_string = base64_string.split(",")[1]


        # Decode base64
        img_bytes = base64.b64decode(base64_string)


        # Convert bytes to numpy array
        img_np = np.frombuffer(
            img_bytes,
            np.uint8
        )


        # Convert numpy array to OpenCV image
        image = cv2.imdecode(
            img_np,
            cv2.IMREAD_COLOR
        )


        if image is None:
            raise ValueError("Invalid image")


        return image


    except Exception as e:

        raise ValueError(
            f"Image decoding failed: {e}"
        )




def preprocess_face_crop(face):

    # Convert BGR → RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )


    # Resize for CNN input
    face = cv2.resize(
        face,
        TARGET_SIZE
    )


    # Normalize pixel values
    face = face.astype(
        np.float32
    ) / 255.0


    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )


    return face

# ============================================================================
# API ROUTES
# ============================================================================


@app.route("/", methods=["GET"])
def index():

    return jsonify({

        "status": "online",

        "service": "CNN Face Detection API",

        "model_loaded": cnn_model is not None,

        "cascade_loaded": not face_cascade.empty(),

        "input_size": "128x128"

    })




@app.route("/predict", methods=["POST"])
def predict():


    if cnn_model is None:

        return jsonify({

            "face_detected": False,

            "error": "Model not loaded"

        }), 500



    try:


        # Get JSON data

        data = request.get_json()



        if not data or "image" not in data:

            return jsonify({

                "face_detected": False,

                "error": "No image received"

            }), 400




        # Decode image

        image = decode_base64_image(
            data["image"]
        )

        # Fix mobile portrait orientation
        height, width = image.shape[:2]

        if height > width:
            image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE
                )


        # Convert to grayscale

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )


        # Improve contrast

        gray = cv2.equalizeHist(gray)




        # Detect faces using Haar Cascade

        faces = face_cascade.detectMultiScale(

            gray,

            scaleFactor=1.05,

            minNeighbors=5,

            minSize=(50,50)

)

        if len(faces) > 0:
            return jsonify({
                "face_detected": True
                })
        else:
            return jsonify({
                "face_detected": False
                })


        # Select largest face

        x, y, w, h = max(

            faces,

            key=lambda f: f[2] * f[3]

        )



        # Add padding

        pad = 10


        x1 = max(
            0,
            x - pad
        )

        y1 = max(
            0,
            y - pad
        )


        x2 = min(
            image.shape[1],
            x + w + pad
        )


        y2 = min(
            image.shape[0],
            y + h + pad
        )



        # Crop face

        face = image[y1:y2, x1:x2]



        if face.size == 0:

            return jsonify({

                "face_detected": False

            }), 200




        # Preprocess face for CNN

        input_tensor = preprocess_face_crop(face)



        # Run CNN prediction

        cnn_model.set_tensor(

            input_details[0]["index"],

            input_tensor.astype(np.float32)

        )


        cnn_model.invoke()



        prediction = cnn_model.get_tensor(

            output_details[0]["index"]

        )



        score = float(
            prediction[0][0]
        )


        print(
            "Prediction Score:",
            score
        )



        # ======================================================
        # CLASSIFICATION
        # 0 = Human Face
        # 1 = Non Human Face
        # ======================================================


        THRESHOLD = 0.70


        face_detected = score < THRESHOLD

        print("CNN SCORE:", score)
        print("FACE RESULT:", face_detected)



        # Return ONLY status
        # No confidence percentage


        return jsonify({

            "face_detected": face_detected

        }), 200




    except Exception as e:


        logging.exception(e)


        return jsonify({

            "face_detected": False,

            "error": str(e)

        }), 500

    # ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "error": "Route not found",

        "message": "Available routes: GET / and POST /predict"

    }), 404




@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "error": "Internal Server Error"

    }), 500




# ============================================================================
# MAIN APPLICATION
# ============================================================================


if __name__ == "__main__":


    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )