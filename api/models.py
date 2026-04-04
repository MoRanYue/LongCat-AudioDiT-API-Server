"""Model management for LongCat-AudioDiT API."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import audiodit  # auto-registers with transformers
from audiodit import AudioDiTModel
import os
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for loading and switching TTS models."""
    
    _instance = None
    _current_model = None
    _current_tokenizer = None
    _current_model_name = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_dir: str, device: str = None) -> Tuple[AudioDiTModel, AutoTokenizer]:
        """Load a TTS model and tokenizer.
        
        Args:
            model_dir: HuggingFace model ID or local path
            device: 'cuda' or 'cpu', defaults to cuda if available
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model from {model_dir} on device {device}")
        
        try:
            # Load model
            model = AudioDiTModel.from_pretrained(model_dir).to(device)
            model.vae.to_half()  # VAE runs in fp16 (matching original)
            model.eval()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
            
            self._current_model = model
            self._current_tokenizer = tokenizer
            self._current_model_name = model_dir
            
            logger.info(f"Model loaded successfully: {model_dir}")
            logger.info(f"Model config: sampling_rate={model.config.sampling_rate}, "
                       f"latent_hop={model.config.latent_hop}, "
                       f"max_wav_duration={model.config.max_wav_duration}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_dir}: {e}")
            raise
    
    def get_model(self) -> Optional[AudioDiTModel]:
        """Get the currently loaded model."""
        return self._current_model
    
    def get_tokenizer(self) -> Optional[AutoTokenizer]:
        """Get the current tokenizer."""
        return self._current_tokenizer
    
    def get_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self._current_model_name
    
    def switch_model(self, model_dir: str, device: str = None) -> bool:
        """Switch to a different model.
        
        Args:
            model_dir: New model directory or HF ID
            device: Device to load on
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.load_model(model_dir, device)
            return True
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._current_model is not None and self._current_tokenizer is not None


class InferenceService:
    """Service for performing TTS inference."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.default_steps = 16
        self.default_guidance_strength = 4.0
        self.default_guidance_method = "cfg"
        
    def load_audio(self, wavpath: str, sr: int = 24000):
        """Load audio file using librosa."""
        import librosa
        audio, _ = librosa.load(wavpath, sr=sr, mono=True)
        return torch.from_numpy(audio).unsqueeze(0)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text (lowercase, remove extra spaces)."""
        import re
        text = text.lower()
        text = re.sub(r'[""\"\'\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def approx_duration_from_text(self, text: str, max_duration: float = 30.0) -> float:
        """Approximate audio duration from text."""
        import re
        EN_DUR_PER_CHAR = 0.082
        ZH_DUR_PER_CHAR = 0.21
        
        text = re.sub(r"\s+", "", text)
        num_zh = num_en = num_other = 0
        for c in text:
            if "\u4e00" <= c <= "\u9fff":
                num_zh += 1
            elif c.isalpha():
                num_en += 1
            else:
                num_other += 1
        
        if num_zh > num_en:
            num_zh += num_other
        else:
            num_en += num_other
        
        return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)
    
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        steps: int = None,
        guidance_strength: float = None,
        guidance_method: str = None,
        seed: int = 1024
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to prompt audio for voice cloning
            prompt_text: Text of the prompt audio
            steps: Number of ODE steps
            guidance_strength: CFG/APG strength
            guidance_method: 'cfg' or 'apg'
            seed: Random seed
            
        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        if not self.model_manager.is_model_loaded():
            raise ValueError("No model loaded. Please load a model first.")
        
        model = self.model_manager.get_model()
        tokenizer = self.model_manager.get_tokenizer()
        
        # Use defaults if not provided
        steps = steps or self.default_steps
        guidance_strength = guidance_strength or self.default_guidance_strength
        guidance_method = guidance_method or self.default_guidance_method
        
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        device = next(model.parameters()).device
        sr = model.config.sampling_rate
        full_hop = model.config.latent_hop
        max_duration = model.config.max_wav_duration
        
        # Normalize text
        text_norm = self.normalize_text(text)
        
        # Handle prompt audio for voice cloning
        if speaker_wav and os.path.exists(speaker_wav):
            no_prompt = False
            prompt_text_norm = self.normalize_text(prompt_text) if prompt_text else ""
            full_text = f"{prompt_text_norm} {text_norm}"
            
            # Load prompt audio
            prompt_wav = self.load_audio(speaker_wav, sr).unsqueeze(0)
            
            # Compute prompt duration
            off = 3
            pw = self.load_audio(speaker_wav, sr)
            if pw.shape[-1] % full_hop != 0:
                pw = F.pad(pw, (0, full_hop - pw.shape[-1] % full_hop))
            pw = F.pad(pw, (0, full_hop * off))
            
            with torch.no_grad():
                plt = model.vae.encode(pw.unsqueeze(0).to(device))
            if off:
                plt = plt[..., :-off]
            prompt_dur = plt.shape[-1]
            
        else:
            no_prompt = True
            full_text = text_norm
            prompt_wav = None
            prompt_dur = 0
        
        # Tokenize
        inputs = tokenizer([full_text], padding="longest", return_tensors="pt")
        
        # Duration estimation
        prompt_time = prompt_dur * full_hop / sr if prompt_dur > 0 else 0
        dur_sec = self.approx_duration_from_text(text_norm, max_duration=max_duration - prompt_time)
        
        if not no_prompt and prompt_text:
            approx_pd = self.approx_duration_from_text(prompt_text_norm, max_duration=max_duration)
            ratio = np.clip(prompt_time / approx_pd, 1.0, 1.5)
            dur_sec = dur_sec * ratio
        
        logger.info(f"Approximate duration: {dur_sec:.3f}s")
        duration = int(dur_sec * sr // full_hop)
        duration = min(duration + prompt_dur, int(max_duration * sr // full_hop))
        
        # Generate
        with torch.no_grad():
            output = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
                prompt_audio=prompt_wav.to(device) if prompt_wav is not None else None,
                duration=duration,
                steps=steps,
                cfg_strength=guidance_strength,
                guidance_method=guidance_method,
            )
        
        wav = output.waveform.squeeze().detach().cpu().numpy()
        return wav, sr
    
    def save_audio(self, audio: np.ndarray, sr: int, filepath: str) -> bool:
        """Save audio to file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            sf.write(filepath, audio, sr)
            return True
        except Exception as e:
            logger.error(f"Failed to save audio to {filepath}: {e}")
            return False