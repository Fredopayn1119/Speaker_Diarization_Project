# This file provides utilities for loading and processing audio files,
# including resampling, normalization, and segmentation.

"""Utilities for loading and processing audio files."""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from typing import Tuple, Optional, Dict, List, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAMPLE_RATE, MONO, NORMALIZE


class AudioProcessor:
    """Class for loading and processing audio files."""
    
    def __init__(self, sample_rate: int = SAMPLE_RATE, mono: bool = MONO, normalize: bool = NORMALIZE):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            mono: Whether to convert audio to mono
            normalize: Whether to normalize audio
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.normalize = normalize
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=self.mono)
            
            if self.normalize:
                audio = librosa.util.normalize(audio)
                
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise
            
    def load_audio_torch(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file using torchaudio.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        try:
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                sr = self.sample_rate
                
            # Convert to mono if needed
            if self.mono and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Normalize if needed
            if self.normalize:
                waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
                
            return waveform, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            raise
            
    def save_audio(self, audio: np.ndarray, file_path: str, sample_rate: Optional[int] = None):
        """
        Save audio data to file.
        
        Args:
            audio: Audio data (numpy array)
            file_path: Path to save audio file
            sample_rate: Sample rate (defaults to self.sample_rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            sf.write(file_path, audio, sample_rate)
        except Exception as e:
            print(f"Error saving audio file {file_path}: {e}")
            raise
            
    def segment_audio(self, audio: np.ndarray, segments: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        Segment audio based on start and end times.
        
        Args:
            audio: Audio data
            segments: List of (start_time, end_time) in seconds
            
        Returns:
            List of audio segments as numpy arrays
        """
        audio_segments = []
        for start_sec, end_sec in segments:
            start_sample = int(start_sec * self.sample_rate)
            end_sample = int(end_sec * self.sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment = audio[start_sample:end_sample]
                audio_segments.append(segment)
                
        return audio_segments
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13, 
                    n_fft: int = 400, hop_length: int = 160) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio data
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length
            
        Returns:
            MFCC features as a numpy array
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs])
        
        return features.T  # Transpose to (time_frames, features)
