# Speaker Diarization Pipeline User Guide

This guide will help you set up and run a complete speaker diarization system with speech recognition (ASR) to identify who spoke when in an audio recording and transcribe what they said.

## 1. Installation & Setup

First, ensure you have all required dependencies:

```bash
# Create a conda environment (recommended)
conda env create -f environment.yml
# Or install requirements via pip
pip install -r requirements.txt
```

### Key Dependencies:

- **RNNoise**: For noise removal - install `librnnoise` or `pyrnnoise`
- **WebRTC VAD**: For voice activity detection
- **Resemblyzer**: For speaker embedding extraction
- **Whisper** or **Transformers**: For speech recognition
- **scikit-learn**: For clustering algorithms
- **librosa & scipy**: For audio processing

## 2. Pipeline Overview

The pipeline consists of five main stages:

1. **Noise Removal**: Cleans audio using RNNoise
2. **Segmentation**: Identifies speech segments using WebRTC VAD and energy-based detection
3. **Feature Extraction**: Extracts speaker embeddings (MFCC, d-vectors, or x-vectors)
4. **Clustering**: Groups segments by speaker (AHC, Spectral, K-means, or DBSCAN)
5. **ASR**: Transcribes speech using Whisper or Transformers models

## 3. Running the Pipeline

Use the `diarization_pipeline.py` script to run the entire process:

```bash
# Basic usage with default parameters
python diarization_pipeline.py --input audio_files/0638.wav

# Specifying more options
python diarization_pipeline.py \
  --input audio_files/0638.wav \
  --output_dir results \
  --embedding dvector \
  --clustering ahc \
  --whisper_model base \
  --visualize
```

### Key Parameters:

- `--input`: Path to input audio file
- `--output_dir`: Directory to save all outputs
- `--embedding`: Speaker embedding method (`mfcc`, `dvector`, or `xvector`)
- `--clustering`: Clustering algorithm (`ahc`, `spectral`, `kmeans`, or `dbscan`)
- `--num_speakers`: Number of speakers (if known)
- `--whisper_model`: ASR model size (`tiny`, `base`, `small`, `medium`, or `large`)
- `--visualize`: Generate visualizations of embeddings and clusters
- `--skip_denoise`, `--skip_segmentation`, `--skip_asr`: Skip specific pipeline stages

## 4. Step-by-Step Verification

To verify each step in the pipeline works correctly:

### 4.1. Noise Removal

```bash
python scripts/noise_removal.py
```

Verify that `audio_files/denoised.wav` is created and has reduced noise compared to the original.

### 4.2. Segmentation

```bash
python scripts/segmentation.py
```

Verify that multiple WAV files are created in `audio_files/segments/` folder, each containing speech.

### 4.3. Feature Extraction

```bash
python scripts/feature_extraction.py
```

Verify that `audio_files/segment_embeddings.npy` is created and embedding visualizations show clusters.

### 4.4. Clustering

```bash
python scripts/clustering.py
```

Verify that `audio_files/diarization_result.json` is created and cluster visualizations show distinct speakers.

### 4.5. ASR Transcription

```bash
python scripts/asr.py
```

Verify that `audio_files/full_result.json` and `audio_files/full_result.txt` are created with transcribed text.

## 5. Output Files

The pipeline generates several output files:

- `denoised.wav`: Cleaned audio file
- `segments/*.wav`: Individual speech segments
- `segment_embeddings.npy`: Speaker embeddings for each segment
- `*_visualization.png`: Visualizations of embeddings and clusters
- `diarization_result.json`: Speaker diarization results (who spoke when)
- `full_result.json`: Complete diarization with transcription
- `full_result.txt`: Human-readable transcription with speaker labels and timestamps

## 6. Visualization and Analysis

To visualize the diarization results, use the Jupyter notebook:

```bash
jupyter notebook notebooks/visualize_diarization.ipynb
```

This notebook allows you to:
- Plot speaker segments over time
- Analyze speaker turn patterns
- View transcriptions by speaker
- Compare different embedding and clustering methods

## 7. Advanced Configuration

### Embedding Methods:

- **MFCC**: Classic features, lightweight but less accurate
- **d-vector**: Uses Resemblyzer's VoiceEncoder, good balance of accuracy and speed
- **x-vector**: Deep learning features, most accurate but computation-intensive

### Clustering Methods:

- **AHC**: Agglomerative Hierarchical Clustering (default), reliable for most cases
- **Spectral**: Better for complex speaker patterns but slower
- **K-means**: Fast but requires knowing the number of speakers
- **DBSCAN**: Automatic cluster detection but needs parameter tuning

## 8. Troubleshooting

1. **RNNoise/denoise command not found**: 
   - Install RNNoise: `pip install pyrnnoise`
   - Or use `--skip_denoise` to skip this step

2. **Error during segmentation**:
   - Ensure WebRTC VAD is installed: `pip install webrtcvad`
   - Check if the input audio format is supported (WAV, 16-bit PCM)

3. **Resemblyzer/VoiceEncoder errors**:
   - Install with: `pip install resemblyzer`
   - Try `--embedding mfcc` if issues persist

4. **Whisper model errors**:
   - Install with: `pip install openai-whisper`
   - Try smaller models (`tiny` or `base`) if memory issues occur
   - Alternatively, install transformers: `pip install transformers`

5. **Segmentation produces too many/few segments**:
   - Adjust the VAD aggressiveness and energy threshold in `segmentation.py`

6. **Incorrect number of speakers**:
   - Use `--num_speakers` to manually set the number if known
   - Try different clustering methods

## 9. Performance Tips

1. Use a GPU for faster processing, especially with larger models
2. Set appropriate Whisper model size:
   - `tiny` or `base`: Quick but less accurate
   - `small`: Good balance for most tasks
   - `medium` or `large`: Best quality but much slower
3. For processing long files, consider breaking them into chunks
4. Use `--skip_denoise` or `--skip_segmentation` if you already have processed files
5. For Mac users with Apple Silicon, the code automatically uses MPS acceleration

This implementation allows you to experiment with different embedding and clustering methods to find the optimal solution for your specific audio characteristics. Remember to adjust parameters based on your audio quality, the number of speakers, and computational resources available.