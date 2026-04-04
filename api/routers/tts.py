"""TTS endpoints for LongCat-AudioDiT API."""

import logging
import io
import os
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, FileResponse
import numpy as np

from ..schemas import (
    SynthesisRequest, SynthesisFileRequest, SuccessResponse
)
from ..config import ConfigManager
from ..models import ModelManager, InferenceService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_config_manager():
    """Dependency for getting ConfigManager instance."""
    from ..app import config_manager
    return config_manager


def get_inference_service():
    """Dependency for getting InferenceService instance."""
    from ..app import inference_service
    return inference_service


@router.get("/tts_stream")
async def tts_stream(
    text: str = Query(..., description="Text to synthesize"),
    speaker_wav: str = Query(..., description="Path to speaker audio for voice cloning"),
    language: str = Query(..., description="Language code (ignored for this model)"),
    config: ConfigManager = Depends(get_config_manager),
    inference: InferenceService = Depends(get_inference_service)
):
    """TTS streaming endpoint (not actually streaming - returns complete audio).
    
    Note: This model doesn't support true streaming, so we return the complete audio.
    """
    try:
        # Generate audio
        audio, sr = inference.synthesize(
            text=text,
            speaker_wav=speaker_wav if os.path.exists(speaker_wav) else None,
            prompt_text=None,  # No separate prompt text for this endpoint
            steps=config.config.tts_steps,
            guidance_strength=config.config.tts_guidance_strength,
            guidance_method=config.config.tts_guidance_method,
            seed=config.config.tts_seed
        )
        
        # Convert to bytes
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)
        
        # Return as streaming response
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{uuid.uuid4().hex[:8]}.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@router.post("/tts_to_audio/")
async def tts_to_audio(
    request: SynthesisRequest,
    config: ConfigManager = Depends(get_config_manager),
    inference: InferenceService = Depends(get_inference_service)
):
    """Synthesize speech and return audio data."""
    try:
        # Generate audio
        audio, sr = inference.synthesize(
            text=request.text,
            speaker_wav=request.speaker_wav if os.path.exists(request.speaker_wav) else None,
            prompt_text=None,  # No separate prompt text for this endpoint
            steps=config.config.tts_steps,
            guidance_strength=config.config.tts_guidance_strength,
            guidance_method=config.config.tts_guidance_method,
            seed=config.config.tts_seed
        )
        
        # Convert to bytes
        import soundfile as sf
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format='WAV')
        buffer.seek(0)
        
        # Return as streaming response
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{uuid.uuid4().hex[:8]}.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")


@router.post("/tts_to_file", response_model=SuccessResponse)
async def tts_to_file(
    request: SynthesisFileRequest,
    config: ConfigManager = Depends(get_config_manager),
    inference: InferenceService = Depends(get_inference_service)
):
    """Synthesize speech and save to file."""
    try:
        # Generate audio
        audio, sr = inference.synthesize(
            text=request.text,
            speaker_wav=request.speaker_wav if os.path.exists(request.speaker_wav) else None,
            prompt_text=None,  # No separate prompt text for this endpoint
            steps=config.config.tts_steps,
            guidance_strength=config.config.tts_guidance_strength,
            guidance_method=config.config.tts_guidance_method,
            seed=config.config.tts_seed
        )
        
        # Determine output path
        if os.path.isabs(request.file_name_or_path):
            output_path = request.file_name_or_path
        else:
            # Check if it's just a filename
            if '/' not in request.file_name_or_path and '\\' not in request.file_name_or_path:
                output_path = config.get_output_path(request.file_name_or_path)
            else:
                # It's a relative path with directories
                output_path = os.path.join(config.config.output_dir, request.file_name_or_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save audio
        success = inference.save_audio(audio, sr, output_path)
        
        if success:
            return SuccessResponse(
                message=f"Audio saved to {output_path}",
                success=True
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save audio file")
        
    except Exception as e:
        logger.error(f"TTS to file failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS to file failed: {str(e)}")