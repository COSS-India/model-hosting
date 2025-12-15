#!/usr/bin/env python3
"""
Custom MLflow ASR Server with /asr endpoint
"""
import os
import sys
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Load MLflow model
MODEL_PATH = os.environ.get("MODEL_PATH", "mlruns/0/models/m-8f33614a5aeb46f6a4f4c8b0c64b9cf7/artifacts")

app = FastAPI(
    title="MLflow ASR Service",
    description="Automatic Speech Recognition service using MLflow",
    version="1.0.0"
)

# Global model variable
model = None

class ASRRequest(BaseModel):
    audio_base64: str
    lang: Optional[str] = "hi"
    decoding: Optional[str] = "ctc"

class ASRResponse(BaseModel):
    transcription: str
    lang: str
    decoding: str

@app.on_event("startup")
async def load_model():
    """Load MLflow model on startup"""
    global model
    try:
        print(f"Loading MLflow model from: {MODEL_PATH}")
        model = mlflow.pyfunc.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "Service Unavailable", "message": "Model not loaded"}
        )
    return {"status": "healthy", "message": "Service is running"}

@app.post("/asr", response_model=ASRResponse)
async def transcribe(request: ASRRequest):
    """
    Transcribe audio to text
    
    Args:
        request: ASRRequest containing audio_base64, lang, and decoding
    
    Returns:
        ASRResponse with transcription, lang, and decoding
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import pandas as pd
        
        # Prepare input in MLflow format
        model_input = pd.DataFrame([{
            "audio_base64": request.audio_base64,
            "lang": request.lang,
            "decoding": request.decoding
        }])
        
        # Get prediction
        predictions = model.predict(model_input)
        
        # Extract transcription
        if isinstance(predictions, pd.DataFrame):
            transcription = predictions.iloc[0]["transcription"]
        else:
            transcription = predictions[0]["transcription"] if isinstance(predictions, list) else str(predictions)
        
        return ASRResponse(
            transcription=transcription,
            lang=request.lang,
            decoding=request.decoding
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing request: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MLflow ASR Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "asr": "/asr",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False
    )






