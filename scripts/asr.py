import os
import numpy as np
import json
import torch
import librosa
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class ASRProcessor:
    """
    Transcribe audio segments using Whisper (OpenAI)
    """
    
    def __init__(self, model_name="tiny", device=None):
        """
        Initialize the ASR processor.
        
        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to run the model on ("cpu", "cuda")
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device} for ASR")
        
        # Load model lazily (on first use)
        self.model = None
    
    def load_model(self):
        """Load the Whisper model if not already loaded"""
        if self.model is not None:
            return
        
        try:
            import whisper
            print(f"Loading Whisper {self.model_name} model...")
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"Whisper {self.model_name} model loaded successfully")
        except ImportError:
            raise ImportError("Whisper is not available. Install with: pip install openai-whisper")
    
    def transcribe_audio(self, audio_path, language=None):
        """
        Transcribe an audio file
        
        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., "en") - if None, will be detected
            
        Returns:
            Transcription result dictionary
        """
        self.load_model()
        
        # Use Whisper
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            word_timestamps=True
        )
        
        return result
    
    def transcribe_segments(self, segment_dir, diarization_result=None):
        """
        Transcribe all audio segments in a directory
        
        Args:
            segment_dir: Directory containing audio segments
            diarization_result: Optional diarization result to match segments with speakers
            
        Returns:
            List of transcribed segments with timing and speaker information
        """
        # Either get segment files from directory or from diarization result
        if diarization_result:
            # Extract all segment filenames from diarization result
            segment_files = []
            segment_info = {}
            
            for speaker_id, segments in diarization_result["speaker_segments"].items():
                for segment in segments:
                    segment_name = segment["segment"]
                    # Handle merged segments (contains '+')
                    if '+' in segment_name:
                        segment_files_parts = segment_name.split('+')
                        segment_files.extend(segment_files_parts)
                        
                        # Store info for each segment
                        for part in segment_files_parts:
                            segment_info[part] = {
                                "speaker_id": speaker_id,
                                "start": segment["start"],
                                "end": segment["end"]
                            }
                    else:
                        segment_files.append(segment_name)
                        segment_info[segment_name] = {
                            "speaker_id": speaker_id,
                            "start": segment["start"],
                            "end": segment["end"]
                        }
        else:
            # Get all wav files in the directory
            segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.wav')])
            segment_info = {}
        
        # Make paths absolute if needed
        segment_files = [f if os.path.isabs(f) else os.path.join(segment_dir, f) for f in segment_files]
        
        # Remove duplicates while maintaining order
        segment_files = list(dict.fromkeys(segment_files))
        
        # Initialize result
        transcribed_segments = []
        
        # Process each segment
        print(f"Transcribing {len(segment_files)} audio segments...")
        for segment_file in tqdm(segment_files):
            segment_basename = os.path.basename(segment_file)
            segment_path = os.path.join(segment_dir, segment_basename) if not os.path.isabs(segment_file) else segment_file
            
            # Skip if file doesn't exist
            if not os.path.exists(segment_path):
                print(f"Warning: Segment file {segment_path} not found, skipping.")
                continue
            
            # Get segment info
            if segment_basename in segment_info:
                info = segment_info[segment_basename]
                speaker_id = info["speaker_id"]
                start_time = info["start"]
                end_time = info["end"]
            else:
                # Try to extract segment number and derive timing
                import re
                match = re.search(r'segment_(\d+)', segment_basename)
                if match:
                    segment_num = int(match.group(1))
                    # Assuming 1-second segments
                    segment_length = 1.0
                    start_time = (segment_num - 1) * segment_length
                    end_time = segment_num * segment_length
                    speaker_id = "unknown"
                else:
                    # Fallback
                    start_time = 0
                    end_time = 1
                    speaker_id = "unknown"
            
            # Skip very short segments
            audio, sr = librosa.load(segment_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            if duration < 0.1:  # Skip very short segments
                continue
                
            try:
                # Transcribe the segment
                result = self.transcribe_audio(segment_path)
                
                # Get the transcribed text
                text = result["text"].strip()
                
                # If empty, skip
                if not text:
                    continue
                
                # Add to result
                transcribed_segments.append({
                    "segment": segment_basename,
                    "speaker_id": speaker_id,
                    "start": start_time,
                    "end": end_time,
                    "text": text
                })
                
            except Exception as e:
                print(f"Error transcribing {segment_basename}: {str(e)}")
        
        # Sort segments by start time
        transcribed_segments.sort(key=lambda x: x["start"])
        
        return transcribed_segments
    
    def integrate_with_diarization(self, transcribed_segments, diarization_result):
        """
        Integrate transcription with diarization results
        
        Args:
            transcribed_segments: List of transcribed segments
            diarization_result: Diarization result dictionary
            
        Returns:
            Updated diarization result with transcriptions
        """
        # Create a copy of the diarization result
        result = diarization_result.copy()
        
        # Add transcribed_segments field
        result["transcribed_segments"] = transcribed_segments
        
        # Add transcriptions to speaker segments
        for speaker_id, segments in result["speaker_segments"].items():
            for segment in segments:
                # Find matching transcribed segments
                transcript = ""
                for trans in transcribed_segments:
                    # Check if this transcription belongs to this segment
                    if trans["speaker_id"] == speaker_id and \
                       ((trans["start"] >= segment["start"] and trans["start"] < segment["end"]) or \
                        (trans["end"] > segment["start"] and trans["end"] <= segment["end"])):
                        if transcript:
                            transcript += " "
                        transcript += trans["text"]
                
                # Add transcription to segment
                segment["transcription"] = transcript
        
        return result
    
    def save_transcription_result(self, result, output_path):
        """
        Save transcription and diarization results to a JSON file
        
        Args:
            result: Combined result dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # Also save a human-readable text version
        text_path = output_path.replace('.json', '.txt')
        with open(text_path, 'w') as f:
            f.write(f"Transcription with {result['num_speakers']} speakers\n")
            f.write("="*50 + "\n\n")
            
            # Sort all segments by time
            all_segments = []
            for speaker_id, segments in result["speaker_segments"].items():
                for segment in segments:
                    if "transcription" in segment and segment["transcription"].strip():
                        all_segments.append({
                            "speaker_id": speaker_id,
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["transcription"]
                        })
            
            # Sort and write
            all_segments.sort(key=lambda x: x["start"])
            for segment in all_segments:
                time_str = f"[{int(segment['start']//60):02d}:{segment['start']%60:06.3f} -> {int(segment['end']//60):02d}:{segment['end']%60:06.3f}]"
                f.write(f"{time_str} Speaker {segment['speaker_id']}: {segment['text']}\n\n")
        
        print(f"Text version saved to {text_path}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio segments")
    parser.add_argument("--segment_dir", default="../audio_files/segments", help="Directory containing audio segments")
    parser.add_argument("--diarization_path", default="../audio_files/diarization_result.json", help="Path to diarization result JSON")
    parser.add_argument("--output_path", default="../audio_files/full_result.json", help="Path to save output JSON")
    parser.add_argument("--model", default="tiny", help="Whisper model name (tiny, base, small, medium, large)")
    
    args = parser.parse_args()
    
    # Load diarization result if exists
    diarization_result = None
    if os.path.exists(args.diarization_path):
        with open(args.diarization_path, 'r') as f:
            diarization_result = json.load(f)
    
    # Create ASR processor
    asr = ASRProcessor(model_name=args.model)
    
    # Transcribe segments
    transcribed_segments = asr.transcribe_segments(args.segment_dir, diarization_result)
    
    # Combine with diarization
    if diarization_result:
        result = asr.integrate_with_diarization(transcribed_segments, diarization_result)
    else:
        # Create a simple result with just transcriptions
        result = {
            "num_speakers": 1,
            "speaker_segments": {"0": []},
            "transcribed_segments": transcribed_segments
        }
    
    # Save results
    asr.save_transcription_result(result, args.output_path)
