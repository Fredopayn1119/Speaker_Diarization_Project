#!/usr/bin/env python3
"""
Speaker Diarization Web App Launcher

This script launches the Speaker Diarization and Transcription Web App.
Simply run this script to start the web interface.

Usage:
    python run_webapp.py
"""

from diarization_pipeline import run_webapp

if __name__ == "__main__":
    print("Starting Speaker Diarization and Transcription Web App...")
    print("Once the app is running, you can upload audio files (.wav or .mp3)")
    print("and perform speaker diarization with transcription.")
    print("\nThe web interface will open in your browser shortly...")
    run_webapp()