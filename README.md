# Pneumonia Detection App

This project is a Pneumonia Detection Web App powered by a deep learning model (EfficientNet-B3). It uses FastAPI for the backend and provides predictions based on chest X-ray images.

## Features

- EfficientNet-B3 model for pneumonia detection
- ONNX runtime support (for Jetson Nano)
- FastAPI backend
- Easy-to-use REST API

## How to Run

1. Clone the repository
2. Install dependencies
3. Start FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

Send a POST request to `/predict/` with an image file.

## Deployment

Tested on Jetson Nano and local PC.

## License

This project is licensed under the MIT License.
