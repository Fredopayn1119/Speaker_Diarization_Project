# This file handles speaker embedding extraction from audio segments.

"""Speaker embedding extraction for speaker diarization."""

import os
import numpy as np
import torch
import torchaudio
from typing import List, Union, Optional, Tuple
from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EMBEDDING_MODEL_TYPE, EMBEDDING_MODEL_PATH, 
    SAMPLE_RATE, DEVICE
)


class SpeakerEmbeddingExtractor(ABC):
    """Base class for speaker embedding extractors."""
    
    @abstractmethod
    def extract_embedding(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract speaker embedding from audio segment.
        
        Args:
            audio: Audio segment (numpy array or torch tensor)
            
        Returns:
            Speaker embedding as numpy array
        """
        pass
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length (L2 normalization).
        
        Args:
            embedding: Speaker embedding
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def extract_embeddings_from_segments(self, 
                                        audio: np.ndarray, 
                                        segments: List[Tuple[float, float]],
                                        sample_rate: int = SAMPLE_RATE) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
        """
        Extract embeddings from multiple segments of audio.
        
        Args:
            audio: Full audio data
            segments: List of (start_time, end_time) tuples in seconds
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (list of embeddings, list of segments)
        """
        embeddings = []
        valid_segments = []
        
        for start, end in segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Ensure indices are within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample - start_sample > 0:
                segment_audio = audio[start_sample:end_sample]
                try:
                    embedding = self.extract_embedding(segment_audio)
                    embeddings.append(embedding)
                    valid_segments.append((start, end))
                except Exception as e:
                    print(f"Error extracting embedding for segment {start:.2f}-{end:.2f}s: {e}")
        
        return embeddings, valid_segments


class XVectorExtractor(SpeakerEmbeddingExtractor):
    """X-Vector speaker embedding extractor."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize X-Vector extractor.
        
        Args:
            model_path: Path to pretrained model
        """
        try:
            # In a full implementation, you would load the X-Vector model here
            # For simplicity, we'll use SpeechBrain's pretrained model
            from speechbrain.pretrained import EncoderClassifier
            
            if model_path is None:
                # Use SpeechBrain's pretrained model
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/embedding/speechbrain_xvector"
                )
            else:
                # Load from specified path
                self.model = torch.load(model_path)
                
            self.model.to(DEVICE)
            self.model.eval()
        except ImportError:
            raise ImportError("SpeechBrain not installed. Install it with: pip install speechbrain")
    
    def extract_embedding(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract X-Vector embedding from audio segment.
        
        Args:
            audio: Audio segment
            
        Returns:
            X-Vector embedding
        """
        with torch.no_grad():
            # Convert to torch tensor if needed
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            else:
                audio_tensor = audio.unsqueeze(0) if audio.dim() == 1 else audio
                
            # Move to device
            audio_tensor = audio_tensor.to(DEVICE)
            
            # Extract embedding
            embedding = self.model.encode_batch(audio_tensor).squeeze().cpu().numpy()
            
            # Normalize embedding
            embedding = self.normalize_embedding(embedding)
            
            return embedding


class DVectorExtractor(SpeakerEmbeddingExtractor):
    """D-Vector speaker embedding extractor using Resemblyzer."""
    
    def __init__(self):
        """Initialize D-Vector extractor using Resemblyzer."""
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder(device=DEVICE)
        except ImportError:
            raise ImportError("Resemblyzer not installed. Install it with: pip install resemblyzer")
    
    def extract_embedding(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract D-Vector embedding from audio segment.
        
        Args:
            audio: Audio segment
            
        Returns:
            D-Vector embedding
        """
        # Convert to numpy if needed
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # Extract embedding
        embedding = self.encoder.embed_utterance(audio)
        
        # Resemblyzer already normalizes the embedding, but we'll normalize again to be sure
        embedding = self.normalize_embedding(embedding)
        
        return embedding


class Wav2Vec2Extractor(SpeakerEmbeddingExtractor):
    """Wav2Vec2 embedding extractor for speaker recognition."""
    
    def __init__(self, model_name: str = "facebook/wav2vec2-base"):
        """
        Initialize Wav2Vec2 extractor.
        
        Args:
            model_name: Model name or path
        """
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            
            # Load model and processor
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2Model.from_pretrained(model_name)
            
            self.model.to(DEVICE)
            self.model.eval()
        except ImportError:
            raise ImportError("Transformers not installed. Install it with: pip install transformers")
    
    def extract_embedding(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract Wav2Vec2 embedding from audio segment.
        
        Args:
            audio: Audio segment
            
        Returns:
            Wav2Vec2 embedding
        """
        with torch.no_grad():
            # Convert to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
                
            # Process audio
            inputs = self.processor(
                audio, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Extract features
            outputs = self.model(**inputs)
            
            # Mean pooling over time dimension to get fixed-length embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize embedding
            embedding = self.normalize_embedding(embedding)
            
            return embedding


def get_embedding_extractor(model_type: str = EMBEDDING_MODEL_TYPE) -> SpeakerEmbeddingExtractor:
    """
    Factory function to get a speaker embedding extractor.
    
    Args:
        model_type: Type of embedding model ("xvector", "dvector", "wav2vec2")
        
    Returns:
        SpeakerEmbeddingExtractor instance
    """
    if model_type == "xvector":
        return XVectorExtractor(EMBEDDING_MODEL_PATH)
    elif model_type == "dvector":
        return DVectorExtractor()
    elif model_type == "wav2vec2":
        return Wav2Vec2Extractor()
    else:
        raise ValueError(f"Unsupported embedding model type: {model_type}")
