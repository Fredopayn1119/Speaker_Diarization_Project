# This file implements Automatic Speech Recognition (ASR) to transcribe audio segments.

"""Automatic Speech Recognition (ASR) module for transcribing speech segments."""

import os
import numpy as np
import torch
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ASR_MODEL_TYPE, ASR_MODEL_SIZE, ASR_LANGUAGE,
    SAMPLE_RATE, DEVICE
)


class BaseASR(ABC):
    """Base class for Automatic Speech Recognition."""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio segment to text.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        pass
    
    def transcribe_segments(self, 
                           audio: np.ndarray, 
                           segments: List[Tuple[float, float]], 
                           sample_rate: int) -> List[str]:
        """
        Transcribe multiple audio segments.
        
        Args:
            audio: Full audio data
            segments: List of (start_time, end_time) tuples in seconds
            sample_rate: Sample rate
            
        Returns:
            List of transcription texts
        """
        transcriptions = []
        
        for start, end in segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample - start_sample > 0:
                segment_audio = audio[start_sample:end_sample]
                try:
                    text = self.transcribe(segment_audio, sample_rate)
                    transcriptions.append(text)
                except Exception as e:
                    print(f"Error transcribing segment {start:.2f}-{end:.2f}s: {e}")
                    transcriptions.append("")
            else:
                transcriptions.append("")
                
        return transcriptions


class WhisperASR(BaseASR):
    """OpenAI Whisper-based ASR."""
    
    def __init__(self, model_size: str = ASR_MODEL_SIZE, language: str = ASR_LANGUAGE):
        """
        Initialize Whisper ASR.
        
        Args:
            model_size: Model size ("tiny", "base", "small", "medium", "large")
            language: Language code (e.g., "en" for English)
        """
        try:
            import whisper
            self.model = whisper.load_model(model_size)
            self.language = language
        except ImportError:
            raise ImportError("Whisper not installed. Install it with: pip install -U openai-whisper")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        # Whisper expects float32 audio normalized between -1 and 1
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            
        # Transcribe
        result = self.model.transcribe(
            audio, 
            language=self.language,
            fp16=torch.cuda.is_available()
        )
        
        return result["text"].strip()


class Wav2Vec2ASR(BaseASR):
    """Wav2Vec2-based ASR."""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base-960h"):
        """
        Initialize Wav2Vec2 ASR.
        
        Args:
            model_name: Model name or path
        """
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            # Load model and processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
            
            self.model.to(DEVICE)
            self.model.eval()
        except ImportError:
            raise ImportError("Transformers not installed. Install it with: pip install transformers")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using Wav2Vec2.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        with torch.no_grad():
            # Resample if needed
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Perform inference
            outputs = self.model(**inputs)
            
            # Convert logits to tokens to text
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()


class HubertASR(BaseASR):
    """HuBERT-based ASR."""
    
    def __init__(self, model_name: str = "facebook/hubert-large-ls960-ft"):
        """
        Initialize HuBERT ASR.
        
        Args:
            model_name: Model name or path
        """
        try:
            from transformers import HubertForCTC, Wav2Vec2Processor
            
            # Load model and processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = HubertForCTC.from_pretrained(model_name)
            
            self.model.to(DEVICE)
            self.model.eval()
        except ImportError:
            raise ImportError("Transformers not installed. Install it with: pip install transformers")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using HuBERT.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
            
        Returns:
            Transcribed text
        """
        with torch.no_grad():
            # Resample if needed
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
                
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Perform inference
            outputs = self.model(**inputs)
            
            # Convert logits to tokens to text
            predicted_ids = torch.argmax(outputs.logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()


def get_asr_model(model_type: str = ASR_MODEL_TYPE) -> BaseASR:
    """
    Factory function to get an ASR model.
    
    Args:
        model_type: Type of ASR model ("whisper", "wav2vec2", "hubert")
        
    Returns:
        BaseASR instance
    """
    if model_type == "whisper":
        return WhisperASR()
    elif model_type == "wav2vec2":
        return Wav2Vec2ASR()
    elif model_type == "hubert":
        return HubertASR()
    else:
        raise ValueError(f"Unsupported ASR model type: {model_type}")
