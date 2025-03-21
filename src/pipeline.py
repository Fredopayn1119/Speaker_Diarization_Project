# This file defines the end-to-end speaker diarization pipeline,
# orchestrating the different modules like VAD, feature extraction, clustering, and ASR.

"""End-to-end diarization pipeline."""

import os
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# Import modules
from data_utils.audio_processor import AudioProcessor
from segmentation.vad import get_vad_model
from feature_extraction.embeddings import get_embedding_extractor
from clustering.ahc import AHCClustering
from transcription.asr import get_asr_model

# Import config
from config import (
    VAD_MODEL_TYPE, EMBEDDING_MODEL_TYPE, 
    CLUSTERING_METHOD, ASR_MODEL_TYPE,
    SAMPLE_RATE
)


@dataclass
class TranscribedSegment:
    """Class to store a transcribed segment."""
    start: float
    end: float
    speaker_id: int
    text: str


class DiarizationResult:
    """Class to store diarization results."""
    
    def __init__(self):
        self.speaker_segments = {}  # speaker_id -> list of segments
        self.transcribed_segments = []  # list of TranscribedSegment objects
        
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "speaker_segments": {
                str(speaker_id): [{"start": s[0], "end": s[1]} for s in segments]
                for speaker_id, segments in self.speaker_segments.items()
            },
            "transcribed_segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker_id": segment.speaker_id,
                    "text": segment.text
                }
                for segment in self.transcribed_segments
            ]
        }
        
    def save_json(self, output_path: str):
        """Save result to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    def save_rttm(self, output_path: str, file_id: str = "audio"):
        """
        Save result in RTTM format.
        
        RTTM format:
        SPEAKER file_id channel_id start_time duration speaker_id <NA> <NA> <NA>
        """
        with open(output_path, 'w') as f:
            for speaker_id, segments in self.speaker_segments.items():
                for start, end in segments:
                    duration = end - start
                    f.write(f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA>\n")
                    
    def save_transcript(self, output_path: str):
        """Save transcript with speaker labels."""
        with open(output_path, 'w') as f:
            for segment in sorted(self.transcribed_segments, key=lambda x: x.start):
                f.write(f"[Speaker {segment.speaker_id}] {segment.text}\n")


class DiarizationPipeline:
    """End-to-end speaker diarization pipeline."""
    
    def __init__(self,
                vad_model_type: str = VAD_MODEL_TYPE,
                embedding_model_type: str = EMBEDDING_MODEL_TYPE,
                clustering_method: str = CLUSTERING_METHOD,
                asr_model_type: str = ASR_MODEL_TYPE):
        """
        Initialize diarization pipeline.
        
        Args:
            vad_model_type: Type of VAD model
            embedding_model_type: Type of embedding model
            clustering_method: Clustering method
            asr_model_type: Type of ASR model
        """
        self.audio_processor = AudioProcessor(sample_rate=SAMPLE_RATE)
        self.vad_model = get_vad_model(vad_model_type)
        self.embedding_extractor = get_embedding_extractor(embedding_model_type)
        self.clusterer = AHCClustering()
        self.asr_model = get_asr_model(asr_model_type)
        
    def process(self, 
               audio_path: str, 
               num_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Process audio file for diarization and transcription.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Number of speakers (if known)
            
        Returns:
            DiarizationResult object
        """
        result = DiarizationResult()
        
        # 1. Load audio
        print("Loading audio...")
        audio, sample_rate = self.audio_processor.load_audio(audio_path)
        
        # 2. Voice Activity Detection
        print("Detecting speech segments...")
        speech_segments = self.vad_model.detect_speech(audio, sample_rate)
        
        if not speech_segments:
            print("No speech detected in audio.")
            return result
            
        # 3. Extract speaker embeddings
        print("Extracting speaker embeddings...")
        embeddings, valid_segments = self.embedding_extractor.extract_embeddings_from_segments(
            audio, speech_segments, sample_rate
        )
        
        if not embeddings:
            print("Failed to extract embeddings.")
            return result
            
        # 4. Cluster embeddings by speaker
        print("Clustering speakers...")
        result.speaker_segments = self.clusterer.cluster_embeddings(
            embeddings, valid_segments, num_speakers
        )
        
        num_speakers_detected = len(result.speaker_segments)
        print(f"Detected {num_speakers_detected} speakers.")
        
        # 5. Transcribe each speaker's segments
        print("Transcribing segments...")
        for speaker_id, segments in result.speaker_segments.items():
            transcriptions = self.asr_model.transcribe_segments(audio, segments, sample_rate)
            
            for (start, end), text in zip(segments, transcriptions):
                if text:  # Only add non-empty transcriptions
                    result.transcribed_segments.append(
                        TranscribedSegment(start, end, speaker_id, text)
                    )
        
        # Sort transcribed segments by start time
        result.transcribed_segments.sort(key=lambda x: x.start)
        
        return result
    
    def process_and_save(self, 
                        audio_path: str, 
                        output_dir: str,
                        num_speakers: Optional[int] = None):
        """
        Process audio file and save results to directory.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save results
            num_speakers: Number of speakers (if known)
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get audio file name without extension
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Process audio
        result = self.process(audio_path, num_speakers)
        
        # Save results
        result.save_json(os.path.join(output_dir, f"{audio_name}_diarization.json"))
        result.save_rttm(os.path.join(output_dir, f"{audio_name}_diarization.rttm"))
        result.save_transcript(os.path.join(output_dir, f"{audio_name}_transcript.txt"))
        
        print(f"Results saved to {output_dir}")
