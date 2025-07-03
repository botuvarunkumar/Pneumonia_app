"""
FastAPI application for Pneumonia Detection using EfficientNet-B3
"""

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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import efficientnet.tfkeras as efn

# Initialize FastAPI app
app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-powered chest X-ray analysis for pneumonia detection using EfficientNet-B3",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for PWA
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
model = None
IMG_SIZE = 224
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = os.getenv("MODEL_PATH", "initial_efficientnet.h5")


def create_efficientnet_model(img_size: int = 224) -> tf.keras.Model:
    """
    Create the EfficientNet-B3 model architecture for pneumonia detection

    Args:
        img_size: Input image size (default: 224 for EfficientNet-B3)

    Returns:
        Compiled Keras model
    """
    base_model = efn.EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model


def load_model():
    """Load the trained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            model = create_efficientnet_model(IMG_SIZE)
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Using uninitialized model - predictions may be random")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = create_efficientnet_model(IMG_SIZE)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model prediction

    Args:
        image: PIL Image object

    Returns:
        Preprocessed image array
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pneumonia Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "web_interface": "/static/index.html"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "loaded" if model is not None else "not_loaded",
        "model_format": "TensorFlow",
        "image_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "classes": CLASS_NAMES
    }


@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict pneumonia from chest X-ray image

    Args:
        file: Uploaded image file (JPEG, PNG, etc.)

    Returns:
        Prediction results with class and confidence
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        prediction_prob = model.predict(processed_image, verbose=0)[0][0]
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
            "model_format": "TensorFlow",
            "processed_image_size": f"{IMG_SIZE}x{IMG_SIZE}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Predict pneumonia for multiple chest X-ray images

    Args:
        files: List of uploaded image files

    Returns:
        Batch prediction results
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = []

    for file in files:
        try:
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File must be an image"
                })
                continue

            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            processed_image = preprocess_image(image)
            prediction_prob = model.predict(processed_image, verbose=0)[0][0]
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
        "model_format": "TensorFlow"
    }


@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if model is None:
        return {"error": "Model not loaded"}

    return {
        "model_type": "EfficientNet-B3",
        "model_format": "TensorFlow",
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
        "classes": CLASS_NAMES,
        "total_params": model.count_params() if hasattr(model, 'count_params') else None
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to False in production
    )