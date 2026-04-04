"""FastAPI application for LongCat-AudioDiT API."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os

from .config import ConfigManager
from .models import ModelManager, InferenceService
from .routers.config import router as config_router
from .routers.tts import router as tts_router

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
config_manager = None
model_manager = None
inference_service = None


def create_app(config: ConfigManager = None) -> FastAPI:
    """Create and configure FastAPI application."""
    global config_manager, model_manager, inference_service
    
    # Use provided config or create default
    config_manager = config or ConfigManager()
    
    # Create model manager and inference service
    model_manager = ModelManager()
    inference_service = InferenceService(model_manager)
    
    # Initialize model on startup
    def load_initial_model():
        try:
            model_manager.load_model(config_manager.config.model_dir)
            logger.info(f"Initial model loaded: {config_manager.config.model_dir}")
        except Exception as e:
            logger.error(f"Failed to load initial model: {e}")
            # Don't raise - allow API to start without model for configuration
    
    # Create FastAPI app
    app = FastAPI(
        title="LongCat-AudioDiT API",
        description="API for LongCat-AudioDiT TTS model",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(config_router, tags=["Configuration"])
    app.include_router(tts_router, tags=["TTS"])
    
    # Health check endpoint
    @app.get("/")
    async def root():
        return {
            "name": "LongCat-AudioDiT API",
            "version": "1.0.0",
            "status": "running",
            "model_loaded": model_manager.is_model_loaded() if model_manager else False
        }
    
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "model_loaded": model_manager.is_model_loaded() if model_manager else False
        }
    
    # Load model after app is created
    @app.on_event("startup")
    async def startup_event():
        load_initial_model()
    
    return app


# Factory function to get instances
def get_config_manager() -> ConfigManager:
    return config_manager


def get_model_manager() -> ModelManager:
    return model_manager


def get_inference_service() -> InferenceService:
    return inference_service