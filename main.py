import os
import io
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import onnxruntime as ort
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize FastAPI app
app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered chest X-ray analysis for pneumonia detection using EfficientNet-B3 ONNX",
    version="1.0.0"
)

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory (for serving HTML/CSS/JS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
ort_session = None
IMG_SIZE = 224
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = "model.onnx"  # Updated to ONNX model path


def load_onnx_model():
    """Load the ONNX model"""
    global ort_session
    try:
        if os.path.exists(MODEL_PATH):
            # Create ONNX Runtime inference session
            # For Jetson Nano, prioritize CPU execution
            providers = ['CPUExecutionProvider']

            # Try CUDA if available (for better performance)
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')

            ort_session = ort.InferenceSession(MODEL_PATH, providers=providers)
            print(f"ONNX model loaded successfully from {MODEL_PATH}")

            # Print model input/output info
            input_info = ort_session.get_inputs()[0]
            output_info = ort_session.get_outputs()[0]
            print(f"Model input shape: {input_info.shape}")
            print(f"Model output shape: {output_info.shape}")
            print(f"Using providers: {ort_session.get_providers()}")

        else:
            print(f"Error: ONNX model file not found at {MODEL_PATH}")

    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        ort_session = None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model input"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Convert to array
        img_array = img_to_array(image)

        # Normalize pixel values to [0,1]
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Convert to float32 (required for ONNX)
        img_array = img_array.astype(np.float32)

        return img_array

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")


def predict_with_onnx(image_array: np.ndarray) -> float:
    """Make prediction using ONNX model"""
    try:
        # Get input name from the model
        input_name = ort_session.get_inputs()[0].name

        # Run inference
        result = ort_session.run(None, {input_name: image_array})

        # Extract prediction probability
        # Adjust indexing based on your model's output shape
        prediction_prob = result[0][0][0] if len(result[0].shape) > 1 else result[0][0]

        return float(prediction_prob)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ONNX inference error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load ONNX model on startup"""
    print("Starting Pneumonia Detection API...")
    load_onnx_model()
    print("Startup complete!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pneumonia Detection API (ONNX)",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": ort_session is not None,
        "model_format": "ONNX",
        "web_interface": "/static/index.html"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "loaded" if ort_session is not None else "not_loaded",
        "model_format": "ONNX",
        "providers": ort_session.get_providers() if ort_session else None,
        "image_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict pneumonia from chest X-ray image using ONNX model

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        Prediction results with class and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Check if model is loaded
    if ort_session is None:
        raise HTTPException(status_code=500, detail="ONNX model not loaded")

    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction using ONNX
        prediction_prob = predict_with_onnx(processed_image)

        # Ensure prediction_prob is between 0 and 1
        prediction_prob = max(0.0, min(1.0, prediction_prob))

        # Determine predicted class and confidence
        predicted_class = CLASS_NAMES[1] if prediction_prob > 0.5 else CLASS_NAMES[0]
        confidence = float(prediction_prob) if prediction_prob > 0.5 else float(1 - prediction_prob)

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "probability": {
                "NORMAL": round(float(1 - prediction_prob), 4),
                "PNEUMONIA": round(float(prediction_prob), 4)
            },
            "filename": file.filename,
            "model_format": "ONNX",
            "processed_image_size": f"{IMG_SIZE}x{IMG_SIZE}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Predict pneumonia for multiple chest X-ray images using ONNX model

    Args:
        files: List of uploaded image files

    Returns:
        Batch prediction results
    """
    if len(files) > 10:  # Limit batch size for Jetson Nano
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    if ort_session is None:
        raise HTTPException(status_code=500, detail="ONNX model not loaded")

    results = []

    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue

            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image)

            # Make prediction using ONNX
            prediction_prob = predict_with_onnx(processed_image)
            prediction_prob = max(0.0, min(1.0, prediction_prob))

            predicted_class = CLASS_NAMES[1] if prediction_prob > 0.5 else CLASS_NAMES[0]
            confidence = float(prediction_prob) if prediction_prob > 0.5 else float(1 - prediction_prob)

            results.append({
                "filename": file.filename,
                "success": True,
                "prediction": predicted_class,
                "confidence": round(confidence, 4),
                "probability": {
                    "NORMAL": round(float(1 - prediction_prob), 4),
                    "PNEUMONIA": round(float(prediction_prob), 4)
                }
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "success": True,
        "total_files": len(files),
        "successful_predictions": len([r for r in results if r.get("success", False)]),
        "results": results,
        "model_format": "ONNX"
    }


@app.get("/model_info")
async def get_model_info():
    """Get ONNX model information"""
    if ort_session is None:
        return {"error": "ONNX model not loaded"}

    try:
        # Get model metadata
        inputs = ort_session.get_inputs()
        outputs = ort_session.get_outputs()

        input_info = {
            "name": inputs[0].name,
            "shape": inputs[0].shape,
            "type": inputs[0].type
        }

        output_info = {
            "name": outputs[0].name,
            "shape": outputs[0].shape,
            "type": outputs[0].type
        }

        return {
            "model_type": "EfficientNet-B3 (ONNX)",
            "model_format": "ONNX",
            "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
            "classes": CLASS_NAMES,
            "input_info": input_info,
            "output_info": output_info,
            "providers": ort_session.get_providers(),
            "model_file": MODEL_PATH
        }

    except Exception as e:
        return {"error": f"Error getting model info: {str(e)}"}


if __name__ == "__main__":
    # Run the API server
    print("Starting Pneumonia Detection API Server...")
    print("Access the web interface at: http://localhost:8000/static/index.html")
    print("API documentation at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",  # Allow external connections
        port=8000,
        reload=False,  # Set to False for production/Jetson Nano
        log_level="info"
    )