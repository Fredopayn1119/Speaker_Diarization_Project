"""Configuration settings for the speaker diarization pipeline."""

# This file contains the configuration parameters for the speaker diarization system,
# such as ASR model size, language, and GPU settings.

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"

# Create directories if they don't exist
for directory in [INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio processing settings
SAMPLE_RATE = 16000
MONO = True
NORMALIZE = True

# Voice Activity Detection (VAD) settings
VAD_MODEL_TYPE = "pyannote"  # Options: "webrtc", "pyannote", "silero"
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_DURATION_MS = 250
VAD_MIN_SILENCE_DURATION_MS = 500

# Speaker embedding settings
EMBEDDING_MODEL_TYPE = "xvector"  # Options: "xvector", "dvector", "wav2vec2"
EMBEDDING_MODEL_PATH = MODELS_DIR / "embedding" / "xvector.pt"
EMBEDDING_WINDOW_SIZE_SEC = 1.5
EMBEDDING_WINDOW_STEP_SEC = 0.75

# Clustering settings
CLUSTERING_METHOD = "ahc"  # Options: "ahc", "spectral", "kmeans", "dbscan"
# For AHC (Agglomerative Hierarchical Clustering)
AHC_DISTANCE_THRESHOLD = 0.85  # Adjust to control number of speakers
MAX_NUM_SPEAKERS = 10
MIN_NUM_SPEAKERS = 1
# For spectral clustering
SPECTRAL_MAX_SPEAKERS = 10

# ASR (Automatic Speech Recognition) settings
ASR_MODEL_TYPE = "whisper"  # Options: "whisper", "wav2vec2", "hubert"
ASR_MODEL_SIZE = "base"  # For Whisper: "tiny", "base", "small", "medium", "large"
ASR_LANGUAGE = "en"  # Language code

# GPU settings
USE_MPS = torch.backends.mps.is_available()
USE_GPU = torch.cuda.is_available()
DEVICE = "mps" if USE_MPS else "cuda" if USE_GPU else "cpu"
