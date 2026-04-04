"""Configuration management for LongCat-AudioDiT API."""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration."""
    output_dir: str = "./output"
    samples_dir: str = "./samples"
    model_dir: str = "meituan-longcat/LongCat-AudioDiT-1B"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    
    # TTS settings
    tts_steps: int = 16
    tts_guidance_strength: float = 4.0
    tts_guidance_method: str = "cfg"  # "cfg" or "apg"
    tts_seed: int = 1024
    
    # Settings that are defined in openapi.json but not used by this model
    tts_stream_chunk_size: int = 0
    tts_temperature: float = 1.0
    tts_speed: float = 1.0
    tts_length_penalty: float = 1.0
    tts_repetition_penalty: float = 1.0
    tts_top_p: float = 1.0
    tts_top_k: int = 0
    tts_enable_text_splitting: bool = False


class ConfigManager:
    """Manager for application configuration."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.config = AppConfig()
        
        # Load config if file exists
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        else:
            # Create default directories
            self._create_default_dirs()
    
    def _create_default_dirs(self):
        """Create default directories if they don't exist."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.samples_dir, exist_ok=True)
        logger.info(f"Created directories: {self.config.output_dir}, {self.config.samples_dir}")
    
    def load_config(self, config_file: str) -> bool:
        """Load configuration from file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update config with loaded data
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            self.config_file = config_file
            self._create_default_dirs()
            logger.info(f"Loaded configuration from {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            return False
    
    def save_config(self, config_file: str = None) -> bool:
        """Save configuration to file."""
        try:
            save_path = config_file or self.config_file
            if not save_path:
                logger.error("No config file specified for saving")
                return False
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            return False
    
    def update_config(self, **kwargs) -> bool:
        """Update configuration with new values."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Ignoring unknown config key: {key}")
            
            # Recreate directories if output or samples dir changed
            if 'output_dir' in kwargs or 'samples_dir' in kwargs:
                self._create_default_dirs()
            
            # Save if config file exists
            if self.config_file:
                self.save_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return asdict(self.config)
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """Get list of available models."""
        return [
            {
                "name": "LongCat-AudioDiT-1B",
                "description": "1B parameter model",
                "hf_id": "meituan-longcat/LongCat-AudioDiT-1B"
            },
            {
                "name": "LongCat-AudioDiT-3.5B",
                "description": "3.5B parameter model (larger)",
                "hf_id": "meituan-longcat/LongCat-AudioDiT-3.5B"
            }
        ]
    
    def get_speakers(self) -> List[Dict[str, str]]:
        """Get list of speakers from samples directory."""
        speakers = []
        
        if os.path.exists(self.config.samples_dir):
            for file in os.listdir(self.config.samples_dir):
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    speaker_name = os.path.splitext(file)[0]
                    speakers.append({
                        "name": speaker_name,
                        "file_path": os.path.join(self.config.samples_dir, file),
                        "voice_id": speaker_name
                    })
        
        return speakers
    
    def get_languages(self) -> List[Dict[str, Any]]:
        """Get list of supported languages."""
        return [
            {"code": "zh-CN", "name": "Chinese (Simplified)", "supported": True},
            {"code": "en-US", "name": "English (US)", "supported": True},
            {"code": "ja-JP", "name": "Japanese", "supported": False},
            {"code": "ko-KR", "name": "Korean", "supported": False},
            {"code": "fr-FR", "name": "French", "supported": False},
            {"code": "de-DE", "name": "German", "supported": False},
            {"code": "es-ES", "name": "Spanish", "supported": False},
        ]
    
    def get_sample_file(self, file_name: str) -> Optional[str]:
        """Get path to a sample file."""
        file_path = os.path.join(self.config.samples_dir, file_name)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return file_path
        return None
    
    def get_output_path(self, filename: str) -> str:
        """Get full output path for a filename."""
        return os.path.join(self.config.output_dir, filename)