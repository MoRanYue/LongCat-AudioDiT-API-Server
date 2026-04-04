"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict, Union


class HTTPValidationError(BaseModel):
    detail: Optional[List[Any]] = None


class ValidationError(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class ModelNameRequest(BaseModel):
    model_name: str = Field(..., description="Model name to switch to")


class OutputFolderRequest(BaseModel):
    output_folder: str = Field(..., description="Output folder path")


class SpeakerFolderRequest(BaseModel):
    speaker_folder: str = Field(..., description="Speaker folder path")


class SynthesisRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    speaker_wav: str = Field(..., description="Path to speaker audio for voice cloning")
    language: str = Field(..., description="Language code (ignored for this model)")


class SynthesisFileRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    speaker_wav: str = Field(..., description="Path to speaker audio for voice cloning")
    language: str = Field(..., description="Language code (ignored for this model)")
    file_name_or_path: str = Field(..., description="Output file name or path")


class TTSSettingsRequest(BaseModel):
    stream_chunk_size: int = Field(..., description="Stream chunk size")
    temperature: float = Field(..., description="Temperature")
    speed: float = Field(..., description="Speed")
    length_penalty: float = Field(..., description="Length penalty")
    repetition_penalty: float = Field(..., description="Repetition penalty")
    top_p: float = Field(..., description="Top-p sampling")
    top_k: int = Field(..., description="Top-k sampling")
    enable_text_splitting: bool = Field(..., description="Enable text splitting")


class TTSSettingsResponse(BaseModel):
    steps: int = Field(default=16, description="Number of ODE steps")
    guidance_strength: float = Field(default=4.0, description="CFG/APG strength")
    guidance_method: str = Field(default="cfg", description="Guidance method: cfg or apg")
    stream_chunk_size: int = Field(default=0, description="Stream chunk size (not supported)")
    temperature: float = Field(default=1.0, description="Temperature (not used)")
    speed: float = Field(default=1.0, description="Speed (not used)")
    length_penalty: float = Field(default=1.0, description="Length penalty (not used)")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty (not used)")
    top_p: float = Field(default=1.0, description="Top-p (not used)")
    top_k: int = Field(default=0, description="Top-k (not used)")
    enable_text_splitting: bool = Field(default=False, description="Text splitting (not used)")


class ModelInfo(BaseModel):
    name: str
    description: str
    hf_id: str


class FolderInfo(BaseModel):
    output_folder: str
    samples_folder: str


class SpeakerInfo(BaseModel):
    name: str
    file_path: str
    voice_id: str


class LanguageInfo(BaseModel):
    code: str
    name: str
    supported: bool = False


class SuccessResponse(BaseModel):
    success: bool = True
    message: str = ""