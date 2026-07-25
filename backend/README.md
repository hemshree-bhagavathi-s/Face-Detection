# Face Detection System - Python Flask Backend API

This repository contains the Python Flask Backend API for the **Face Detection System using CNN**. It decodes Base64 video frame images sent from the web frontend, uses OpenCV's Haar Cascade classifier for face detection and region cropping, preprocesses the cropped face tensor, and predicts whether a face is present using a trained Keras CNN model (`face_detector.h5`).

---

## 📁 Directory Structure

```text
backend/
├── app.py                            # Main Flask server application
├── requirements.txt                  # Python dependencies
├── face_detector.h5                  # Trained CNN model file
├── haarcascade_frontalface_default.xml # OpenCV Haar Cascade classifier XML
├── .gitignore                        # Python gitignore configuration
└── README.md                         # Documentation & deployment guide
```

---

## 🛠️ Step-by-Step Installation & Setup

### Prerequisites
- **Python 3.11** installed on your system.
- `pip` package manager.

### 1. Create a Virtual Environment
Navigate to the `backend/` directory and create a virtual environment:

#### On Windows:
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

#### On macOS / Linux:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

---

### 2. Install Dependencies
Install all required Python packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Run the Flask API Server
Start the Flask development server:

```bash
python app.py
```

The server will launch on:
- **Local URL**: `http://127.0.0.1:5000`
- **Prediction Endpoint**: `http://127.0.0.1:5000/predict`

---

## 📑 API Documentation

### 1. Health Check Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**:
```json
{
  "status": "online",
  "service": "Face Detection System CNN API",
  "model_loaded": true,
  "cascade_loaded": true,
  "input_resolution": "224x224"
}
```

---

### 2. Prediction Endpoint
- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Payload:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

#### Successful Response (Face Detected):
```json
{
  "face_detected": true,
  "confidence": 96.4
}
```

#### Response (No Face Detected):
```json
{
  "face_detected": false,
  "confidence": 0
}
```

#### Error Response Example (Invalid Input):
```json
{
  "error": "Invalid request payload. 'image' field (base64 string) is required.",
  "face_detected": false,
  "confidence": 0
}
```

---

## 🧪 Testing with Postman

1. Open **Postman** and create a new request.
2. Set the HTTP Method to **`POST`**.
3. Enter the URL: `http://127.0.0.1:5000/predict`
4. Click on the **Headers** tab and set:
   - **Key**: `Content-Type`
   - **Value**: `application/json`
5. Click on the **Body** tab, select **raw**, and select **JSON** format.
6. Paste a valid Base64 encoded image JSON body:
   ```json
   {
     "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..."
   }
   ```
7. Click **Send**. Postman will return the `face_detected` status and `confidence` percentage.

---

## 🚀 Deployment Guide (Render)

Deploying the Flask Backend to **Render.com** is straightforward:

1. **Push Backend Code to GitHub**:
   Ensure `app.py`, `requirements.txt`, `face_detector.h5`, and `haarcascade_frontalface_default.xml` are committed to your GitHub repository.

2. **Create New Web Service on Render**:
   - Log into [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** -> **Web Service**.
   - Connect your GitHub repository.

3. **Configure Service Settings**:
   - **Name**: `face-detection-backend`
   - **Environment**: `Python 3`
   - **Root Directory**: `backend` (if in a subfolder) or leave blank if top level.
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     gunicorn app:app
     ```

4. **Deploy**:
   - Click **Create Web Service**.
   - Render will automatically build the container and deploy your Flask API.
   - Copy your Render backend URL (e.g. `https://face-detection-backend.onrender.com/predict`) and paste it into the **API Settings** panel in the web frontend!
