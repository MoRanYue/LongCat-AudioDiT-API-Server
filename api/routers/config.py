"""Configuration and information endpoints for LongCat-AudioDiT API."""

import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import os

from ..schemas import (
    ModelNameRequest, OutputFolderRequest, SpeakerFolderRequest,
    TTSSettingsRequest, TTSSettingsResponse, ModelInfo, FolderInfo,
    SpeakerInfo, LanguageInfo, SuccessResponse
)
from ..config import ConfigManager
from ..models import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_config_manager():
    """Dependency for getting ConfigManager instance."""
    # This will be set up in main app
    from ..app import config_manager
    return config_manager


def get_model_manager():
    """Dependency for getting ModelManager instance."""
    return ModelManager()


@router.get("/speakers_list", response_model=List[SpeakerInfo])
async def get_speakers_list(
    config: ConfigManager = Depends(get_config_manager)
):
    """Get list of available speakers (same as /speakers)."""
    return config.get_speakers()


@router.get("/speakers", response_model=List[SpeakerInfo])
async def get_speakers(
    config: ConfigManager = Depends(get_config_manager)
):
    """Get list of available speakers from samples directory."""
    return config.get_speakers()


@router.get("/languages", response_model=List[LanguageInfo])
async def get_languages(
    config: ConfigManager = Depends(get_config_manager)
):
    """Get list of supported languages."""
    return config.get_languages()


@router.get("/get_folders", response_model=FolderInfo)
async def get_folders(
    config: ConfigManager = Depends(get_config_manager)
):
    """Get current folder configuration."""
    return FolderInfo(
        output_folder=config.config.output_dir,
        samples_folder=config.config.samples_dir
    )


@router.get("/get_models_list", response_model=List[ModelInfo])
async def get_models_list(
    config: ConfigManager = Depends(get_config_manager)
):
    """Get list of available models."""
    return config.get_available_models()


@router.get("/get_tts_settings", response_model=TTSSettingsResponse)
async def get_tts_settings(
    config: ConfigManager = Depends(get_config_manager)
):
    """Get current TTS settings."""
    return TTSSettingsResponse(
        steps=config.config.tts_steps,
        guidance_strength=config.config.tts_guidance_strength,
        guidance_method=config.config.tts_guidance_method,
        stream_chunk_size=config.config.tts_stream_chunk_size,
        temperature=config.config.tts_temperature,
        speed=config.config.tts_speed,
        length_penalty=config.config.tts_length_penalty,
        repetition_penalty=config.config.tts_repetition_penalty,
        top_p=config.config.tts_top_p,
        top_k=config.config.tts_top_k,
        enable_text_splitting=config.config.tts_enable_text_splitting
    )


@router.get("/sample/{file_name}")
async def get_sample(
    file_name: str,
    config: ConfigManager = Depends(get_config_manager)
):
    """Get sample audio file."""
    file_path = config.get_sample_file(file_name)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Sample file '{file_name}' not found")
    
    # Return file response
    from fastapi.responses import FileResponse
    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type='audio/wav'
    )


@router.post("/set_output", response_model=SuccessResponse)
async def set_output(
    request: OutputFolderRequest,
    config: ConfigManager = Depends(get_config_manager)
):
    """Set output folder."""
    success = config.update_config(output_dir=request.output_folder)
    if success:
        return SuccessResponse(message=f"Output folder set to {request.output_folder}")
    else:
        raise HTTPException(status_code=500, detail="Failed to update output folder")


@router.post("/set_speaker_folder", response_model=SuccessResponse)
async def set_speaker_folder(
    request: SpeakerFolderRequest,
    config: ConfigManager = Depends(get_config_manager)
):
    """Set speaker folder."""
    success = config.update_config(samples_dir=request.speaker_folder)
    if success:
        return SuccessResponse(message=f"Speaker folder set to {request.speaker_folder}")
    else:
        raise HTTPException(status_code=500, detail="Failed to update speaker folder")


@router.post("/switch_model", response_model=SuccessResponse)
async def switch_model(
    request: ModelNameRequest,
    config: ConfigManager = Depends(get_config_manager),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Switch to a different model."""
    # Find model HF ID from name
    models = config.get_available_models()
    model_hf_id = None
    
    for model in models:
        if model["name"] == request.model_name:
            model_hf_id = model["hf_id"]
            break
    
    if not model_hf_id:
        # Try to use the provided name directly
        model_hf_id = request.model_name
    
    # Switch model
    success = model_manager.switch_model(model_hf_id)
    if success:
        # Update config
        config.update_config(model_dir=model_hf_id)
        return SuccessResponse(message=f"Switched to model: {request.model_name}")
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to switch to model: {request.model_name}"
        )


@router.post("/set_tts_settings", response_model=SuccessResponse)
async def set_tts_settings(
    request: TTSSettingsRequest,
    config: ConfigManager = Depends(get_config_manager)
):
    """Set TTS settings."""
    # Update config with provided settings
    # Note: only steps, guidance_strength, guidance_method are actually used by the model
    # Other parameters are stored for API compatibility
    
    success = config.update_config(
        tts_steps=16,  # Keep default, model doesn't use stream_chunk_size
        tts_guidance_strength=4.0,  # Keep default
        tts_guidance_method="cfg",  # Keep default
        tts_stream_chunk_size=request.stream_chunk_size,
        tts_temperature=request.temperature,
        tts_speed=request.speed,
        tts_length_penalty=request.length_penalty,
        tts_repetition_penalty=request.repetition_penalty,
        tts_top_p=request.top_p,
        tts_top_k=request.top_k,
        tts_enable_text_splitting=request.enable_text_splitting
    )
    
    if success:
        return SuccessResponse(message="TTS settings updated")
    else:
        raise HTTPException(status_code=500, detail="Failed to update TTS settings")