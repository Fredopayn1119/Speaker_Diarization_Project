#!/usr/bin/env python3
"""
Speaker Diarization Pipeline

This script runs the full speaker diarization pipeline:
1. Noise removal
2. Audio segmentation
3. Feature extraction
4. Speaker clustering
5. Automatic Speech Recognition (ASR)
6. Result visualization and output

Author: Priyanshu Agrawal
"""

import os
import argparse
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
import warnings
warnings.filterwarnings("ignore")

# Import components
from scripts.feature_extraction import FeatureExtractor
from scripts.clustering import SpeakerClustering
from scripts.asr import ASRProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('diarization_pipeline')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Speaker Diarization Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i", 
        type=str,
        default="audio_files/0638.wav",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--output_dir", "-o", 
        type=str,
        default="audio_files",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--skip_denoise", 
        action="store_true",
        help="Skip the denoising step"
    )
    
    parser.add_argument(
        "--skip_segmentation", 
        action="store_true",
        help="Skip segmentation (use existing segments)"
    )
    
    parser.add_argument(
        "--embedding", 
        type=str,
        choices=["mfcc", "dvector", "xvector"],
        default="dvector",
        help="Speaker embedding method"
    )
    
    parser.add_argument(
        "--clustering", 
        type=str,
        choices=["ahc", "spectral", "kmeans", "dbscan"],
        default="ahc",
        help="Clustering method"
    )
    
    parser.add_argument(
        "--num_speakers", 
        type=int,
        default=None,
        help="Number of speakers (if known). If not provided, will be estimated."
    )
    
    parser.add_argument(
        "--whisper_model", 
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="tiny",
        help="Whisper ASR model size"
    )
    
    parser.add_argument(
        "--skip_asr", 
        action="store_true",
        help="Skip the ASR step"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Generate visualizations for embeddings and clusters"
    )
    
    return parser.parse_args()

def ensure_directory_exists(path):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def noise_removal(input_file, output_file, skip_denoise=False):
    """Perform noise removal using the denoise tool"""
    if skip_denoise and os.path.exists(output_file):
        logger.info(f"Skipping denoising, using existing file: {output_file}")
        return output_file
    
    logger.info(f"Removing noise from {input_file}")
    try:
        subprocess.run(['denoise', input_file, output_file, '--plot'], check=True)
        logger.info(f"Denoising completed. Saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during denoising: {e}")
        logger.warning(f"Using original file without denoising: {input_file}")
        return input_file
    except FileNotFoundError:
        logger.error("Denoise command not found. Make sure RNNoise is installed.")
        logger.warning(f"Using original file without denoising: {input_file}")
        return input_file

def segment_audio(input_file, output_dir, skip_segmentation=False):
    """Segment audio into speech chunks"""
    segment_dir = os.path.join(output_dir, "segments")
    ensure_directory_exists(segment_dir)
    
    if skip_segmentation and os.listdir(segment_dir):
        logger.info(f"Skipping segmentation, using existing segments in {segment_dir}")
        return segment_dir
    
    logger.info(f"Segmenting audio from {input_file}")
    
    try:
        # Import and run segmentation script with the input file path
        import sys
        import importlib.util
        
        # Store original argv and sys.path
        original_argv = sys.argv
        original_path = sys.path.copy()
        
        # Get absolute path to segment_dir to pass to segmentation script
        abs_segment_dir = os.path.abspath(segment_dir)
        
        # Modify sys.argv to include the input file path and segments output directory
        sys.argv = [sys.argv[0], input_file, abs_segment_dir]
        
        # Modify sys.path to include the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        
        # Set environment variable for segments directory
        os.environ['SEGMENTS_DIR'] = abs_segment_dir
        
        # Import the segmentation module
        import scripts.segmentation as segmentation
        
        # Reload the module to ensure fresh execution with new paths
        importlib.reload(segmentation)
        
        # Restore original argv and sys.path
        sys.argv = original_argv
        sys.path = original_path
        
        logger.info(f"Segmentation completed. Segments saved to {segment_dir}")
        return segment_dir
        
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        raise

def extract_features(segment_dir, output_dir, embedding_method="dvector", visualize=False):
    """Extract speaker embeddings from segments"""
    output_file = os.path.join(output_dir, "segment_embeddings.npy")
    
    logger.info(f"Extracting {embedding_method} embeddings from segments in {segment_dir}")
    
    # Create feature extractor
    extractor = FeatureExtractor(method=embedding_method)
    
    # Process segments and extract embeddings
    embeddings = extractor.process_segments(segment_dir)
    
    # Save embeddings
    np.save(output_file, embeddings)
    logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")
    
    # Visualize embeddings if requested
    if visualize:
        viz_file = os.path.join(output_dir, f"{embedding_method}_embeddings_visualization.png")
        logger.info(f"Generating embedding visualization: {viz_file}")
        extractor.visualize_embeddings(embeddings, viz_file)
    
    return embeddings

def cluster_speakers(embeddings, output_dir, clustering_method="ahc", num_speakers=None, visualize=False):
    """Cluster segments by speaker"""
    logger.info(f"Clustering segments using {clustering_method} method")
    
    # Create clustering model
    clustering = SpeakerClustering(method=clustering_method)
    
    # Perform clustering
    segment_to_speaker = clustering.perform_clustering(embeddings, num_speakers=num_speakers)
    
    # Generate diarization result
    diarization_result = clustering.generate_diarization_result(segment_to_speaker)
    
    # Save results
    output_file = os.path.join(output_dir, "diarization_result.json")
    clustering.save_diarization_result(diarization_result, output_file)
    logger.info(f"Diarization results saved to {output_file}")
    
    # Visualize clusters if requested
    if visualize:
        viz_file = os.path.join(output_dir, f"{clustering_method}_clustering_visualization.png")
        logger.info(f"Generating cluster visualization: {viz_file}")
        clustering.visualize_clusters(viz_file)
    
    return diarization_result

def transcribe_audio(segment_dir, diarization_result, output_dir, model_name="tiny"):
    """Transcribe audio segments and integrate with diarization"""
    logger.info(f"Transcribing audio segments using Whisper {model_name} model")
    
    # Create ASR processor
    asr = ASRProcessor(model_name=model_name)
    
    # Transcribe segments
    transcribed_segments = asr.transcribe_segments(segment_dir, diarization_result)
    
    # Combine with diarization
    result = asr.integrate_with_diarization(transcribed_segments, diarization_result)
    
    # Save results
    output_file = os.path.join(output_dir, "full_result.json")
    asr.save_transcription_result(result, output_file)
    logger.info(f"Full diarization and transcription results saved to {output_file}")
    
    return result

def main():
    """Main pipeline function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Ensure output directory exists
    ensure_directory_exists(args.output_dir)
    
    # Record start time
    start_time = time.time()
    
    logger.info(f"Starting speaker diarization pipeline for {args.input}")
    
    # Step 1: Noise Removal
    input_file = os.path.abspath(args.input)
    denoised_file = os.path.join(args.output_dir, "denoised.wav")
    cleaned_audio = noise_removal(input_file, denoised_file, args.skip_denoise)
    
    # Step 2: Audio Segmentation
    segment_dir = segment_audio(cleaned_audio, args.output_dir, args.skip_segmentation)
    
    # Step 3: Feature Extraction
    embeddings = extract_features(segment_dir, args.output_dir, args.embedding, args.visualize)
    
    # Step 4: Speaker Clustering
    diarization_result = cluster_speakers(
        embeddings, 
        args.output_dir, 
        args.clustering, 
        args.num_speakers, 
        args.visualize
    )
    
    # Step 5: Automatic Speech Recognition (ASR)
    if not args.skip_asr:
        final_result = transcribe_audio(
            segment_dir, 
            diarization_result, 
            args.output_dir, 
            args.whisper_model
        )
        
        # Print some statistics
        num_speakers = final_result["num_speakers"]
        num_segments = sum(len(segs) for segs in final_result["speaker_segments"].values())
        
        logger.info(f"Completed pipeline. Detected {num_speakers} speakers and {num_segments} segments.")
        
        # Print a sample of the transcription
        text_path = os.path.join(args.output_dir, "full_result.txt")
        logger.info(f"Transcription saved to {text_path}")
        
    else:
        logger.info("Skipped ASR step. Diarization completed without transcription.")
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    logger.info(f"Pipeline completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()