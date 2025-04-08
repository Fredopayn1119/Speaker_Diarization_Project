# Speaker Diarization System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning-based speaker diarization and transcription system that identifies and separates different speakers in an audio recording and provides a transcript of what each speaker said.

![Speaker Diarization Demo](https://i.imgur.com/example-image.png)

## Features

- 🎤 **Speaker Diarization**: Automatically identifies different speakers in audio recordings
- 🔊 **Noise Removal**: Cleans up background noise for better processing
- 📊 **Visualization**: View speaker clustering and segmentation visualizations
- 📝 **Transcription**: Full transcription of audio with speaker labels
- 👤 **Custom Speaker Naming**: Assign real names to identified speakers
- 🌐 **Web Interface**: User-friendly Streamlit web application

## How It Works

The system implements a complete speaker diarization pipeline:

1. **Noise Removal**: Cleans up audio to improve processing accuracy
2. **Audio Segmentation**: Splits audio into speech segments
3. **Feature Extraction**: Extracts speaker embeddings (d-vectors)
4. **Speaker Clustering**: Groups segments by speaker using Agglomerative Hierarchical Clustering (AHC)
5. **Transcription**: Transcribes each segment using Whisper ASR
6. **Result Integration**: Combines diarization and transcription results

## Installation

### Prerequisites

- Python 3.9 or higher
- Conda (for environment management)
- FFmpeg (for audio processing)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/speaker-diarization.git
   cd speaker-diarization
   ```

2. Create a Conda environment:
   ```bash
   conda create -n speaker-diarization python=3.9 -y
   conda activate speaker-diarization
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface

The easiest way to use the system is through the web interface:

```bash
python run_webapp.py
```

This will start a Streamlit web app where you can:
- Upload audio files (.wav or .mp3)
- Adjust clustering parameters
- Generate speaker-diarized transcripts
- Assign custom names to speakers
- Save and download the results

### Command-line Interface

For batch processing or integration into other workflows:

```bash
python diarization_pipeline.py --input your_audio.wav --output_dir results --whisper_model medium
```

#### Options:
- `--input`: Input audio file path
- `--output_dir`: Directory to save results
- `--num_speakers`: Number of speakers (if known)
- `--threshold`: Clustering threshold (0.1-0.5, default: 0.3)
- `--linkage`: Clustering linkage method (average, complete, single, ward)
- `--whisper_model`: ASR model size (tiny, base, small, medium, large)
- `--visualize`: Generate visualizations
- See more options with `python diarization_pipeline.py --help`

## Example Output

```
[Speaker 1]: Good afternoon everyone, welcome to our meeting.
[Speaker 2]: Thanks for having us. I'd like to discuss the project timeline.
[Speaker 1]: Sure, I think we should start with the requirements gathering phase.
[Speaker 3]: I can help with that part. My team is available next week.
```

## Project Structure

```
├── app.py                    # Streamlit web application
├── run_webapp.py             # Script to launch the web app
├── diarization_pipeline.py   # Main pipeline implementation
├── scripts/
│   ├── asr.py                # Speech recognition module
│   ├── clustering.py         # Speaker clustering algorithms
│   ├── feature_extraction.py # Speaker embedding extraction
│   ├── noise_removal.py      # Audio preprocessing
│   └── segmentation.py       # Audio segmentation
├── sample_audio/             # Storage for processed files
└── requirements.txt          # Package dependencies
```

## Requirements

Key dependencies include:
- streamlit
- numpy
- scipy
- librosa
- scikit-learn
- matplotlib
- torch
- transformers
- whisper

See `requirements.txt` for the complete list.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Streamlit for the web interface
- PyTorch for the deep learning framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request