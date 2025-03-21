# This is the main script for processing audio files, running the speaker diarization pipeline,
# and generating diarization and transcription results.

"""Command-line interface for processing audio files."""

import os
import argparse
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline
from src.pipeline import DiarizationPipeline
from src.config import (
    VAD_MODEL_TYPE, EMBEDDING_MODEL_TYPE, 
    CLUSTERING_METHOD, ASR_MODEL_TYPE,
    INPUT_DIR, OUTPUT_DIR
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Speaker diarization and transcription pipeline"
    )
    
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input audio file or directory"
    )
    
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output directory (default: data/output)"
    )
    
    parser.add_argument(
        "--num-speakers", "-n", type=int, default=None,
        help="Number of speakers (if known)"
    )
    
    parser.add_argument(
        "--vad", type=str, default=VAD_MODEL_TYPE,
        choices=["webrtc", "pyannote", "silero"],
        help="Voice Activity Detection model type"
    )
    
    parser.add_argument(
        "--embedding", type=str, default=EMBEDDING_MODEL_TYPE,
        choices=["xvector", "dvector", "wav2vec2"],
        help="Speaker embedding model type"
    )
    
    parser.add_argument(
        "--clustering", type=str, default=CLUSTERING_METHOD,
        choices=["ahc", "spectral", "kmeans", "dbscan"],
        help="Clustering method"
    )
    
    parser.add_argument(
        "--asr", type=str, default=ASR_MODEL_TYPE,
        choices=["whisper", "wav2vec2", "hubert"],
        help="ASR model type"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set output directory
    output_dir = args.output if args.output else OUTPUT_DIR
    
    # Initialize pipeline
    pipeline = DiarizationPipeline(
        vad_model_type=args.vad,
        embedding_model_type=args.embedding,
        clustering_method=args.clustering,
        asr_model_type=args.asr
    )
    
    # Process input
    if os.path.isdir(args.input):
        # Process all audio files in directory
        for filename in os.listdir(args.input):
            if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
                input_path = os.path.join(args.input, filename)
                print(f"Processing {input_path}...")
                pipeline.process_and_save(input_path, output_dir, args.num_speakers)
    else:
        # Process single file
        pipeline.process_and_save(args.input, output_dir, args.num_speakers)


if __name__ == "__main__":
    main()
