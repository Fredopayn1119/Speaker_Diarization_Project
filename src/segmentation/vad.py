# This file implements Voice Activity Detection (VAD) to identify speech segments in audio.

"""Voice Activity Detection (VAD) module for detecting speech segments."""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    VAD_MODEL_TYPE, VAD_THRESHOLD, 
    VAD_MIN_SPEECH_DURATION_MS, VAD_MIN_SILENCE_DURATION_MS,
    SAMPLE_RATE, DEVICE
)


class BaseVAD(ABC):
    """Base class for Voice Activity Detection."""
    
    @abstractmethod
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        pass
    
    def merge_segments(self, segments: List[Tuple[float, float]], 
                      min_speech_duration: float = VAD_MIN_SPEECH_DURATION_MS/1000, 
                      min_silence_duration: float = VAD_MIN_SILENCE_DURATION_MS/1000) -> List[Tuple[float, float]]:
        """
        Merge segments with small silence gaps and remove short segments.
        
        Args:
            segments: List of (start_time, end_time) tuples
            min_speech_duration: Minimum speech segment duration (seconds)
            min_silence_duration: Minimum silence duration to split segments (seconds)
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
            
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        # Merge segments with small silence gaps
        merged_segments = [segments[0]]
        
        for start, end in segments[1:]:
            prev_start, prev_end = merged_segments[-1]
            
            # If the gap is small, merge
            if start - prev_end < min_silence_duration:
                merged_segments[-1] = (prev_start, max(prev_end, end))
            else:
                merged_segments.append((start, end))
                
        # Filter out short segments
        filtered_segments = [
            (start, end) for start, end in merged_segments
            if end - start >= min_speech_duration
        ]
        
        return filtered_segments


class WebRTCVAD(BaseVAD):
    """WebRTC-based Voice Activity Detection."""
    
    def __init__(self, aggressiveness: int = 3, frame_duration_ms: int = 30):
        """
        Initialize WebRTC VAD.
        
        Args:
            aggressiveness: VAD aggressiveness (0-3)
            frame_duration_ms: Frame duration in milliseconds
        """
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
        except ImportError:
            raise ImportError("webrtcvad not installed. Install it with: pip install webrtcvad")
            
        self.frame_duration_ms = frame_duration_ms
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments using WebRTC VAD.
        
        Args:
            audio: Audio data (mono, 16-bit)
            sample_rate: Sample rate (must be 8000, 16000, 32000, or 48000 Hz)
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        # Convert to int16 PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Calculate frame size
        frame_size = int(sample_rate * self.frame_duration_ms / 1000)
        
        # Pad audio to ensure it's a multiple of frame_size
        pad_size = frame_size - (len(audio_int16) % frame_size)
        if pad_size < frame_size:
            audio_int16 = np.pad(audio_int16, (0, pad_size))
            
        # Split audio into frames
        frames = [audio_int16[i:i+frame_size] for i in range(0, len(audio_int16), frame_size)]
        
        # Detect speech in each frame
        speech_frames = []
        for i, frame in enumerate(frames):
            if len(frame) < frame_size:
                continue
                
            is_speech = self.vad.is_speech(frame.tobytes(), sample_rate)
            if is_speech:
                frame_start_time = i * self.frame_duration_ms / 1000
                frame_end_time = (i + 1) * self.frame_duration_ms / 1000
                speech_frames.append((frame_start_time, frame_end_time))
                
        # Merge consecutive speech frames
        segments = self.merge_segments(speech_frames)
        
        return segments


class PyannotePyTorchVAD(BaseVAD):
    """Pyannote-audio based Voice Activity Detection."""
    
    def __init__(self, model_name: str = "pyannote/segmentation", threshold: float = VAD_THRESHOLD):
        """
        Initialize Pyannote VAD.
        
        Args:
            model_name: Model name or path
            threshold: Detection threshold
        """
        self.threshold = threshold
        try:
            # This is a placeholder. In actual use, you would:
            # 1. Import pyannote.audio
            # 2. Load the pretrained model
            # But for now, we'll just check if it's installed
            import pyannote.audio
            self.model_loaded = False  # Will load on first use to avoid slow imports
            self.model_name = model_name
        except ImportError:
            raise ImportError("pyannote.audio not installed. Install it with: pip install pyannote.audio")
            
    def _load_model(self):
        """Load the Pyannote VAD model."""
        if not hasattr(self, 'model') or self.model is None:
            from pyannote.audio import Pipeline
            self.model = Pipeline.from_pretrained(self.model_name, use_auth_token=True)
            self.model_loaded = True
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Detect speech segments using Pyannote VAD.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        self._load_model()
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        # Pyannote expects a dictionary with the audio and the sample rate
        audio_dict = {'waveform': torch.from_numpy(audio).unsqueeze(0), 
                      'sample_rate': sample_rate}
        
        # Get VAD output
        vad_output = self.model(audio_dict)
        
        # Extract speech segments
        segments = []
        for segment, _, score in vad_output.itertracks(yield_score=True):
            if score > self.threshold:
                segments.append((segment.start, segment.end))
                
        # Merge segments
        segments = self.merge_segments(segments)
        
        return segments


def get_vad_model(model_type: str = VAD_MODEL_TYPE) -> BaseVAD:
    """
    Factory function to get a VAD model.
    
    Args:
        model_type: Type of VAD model ("webrtc", "pyannote", "silero")
        
    Returns:
        BaseVAD instance
    """
    if model_type == "webrtc":
        return WebRTCVAD()
    elif model_type == "pyannote":
        return PyannotePyTorchVAD()
    else:
        raise ValueError(f"Unsupported VAD model type: {model_type}")
