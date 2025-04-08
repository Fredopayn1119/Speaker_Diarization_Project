import streamlit as st
import os
import tempfile
import json
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress PyTorch warnings
import torch
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

# Suppress other warnings
logging.getLogger('matplotlib.font_manager').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from diarization_pipeline import noise_removal, segment_audio, extract_features, cluster_speakers, transcribe_audio, ensure_directory_exists
import datetime

st.set_page_config(page_title="Speaker Diarization and Transcription", layout="centered")

st.markdown("""
    <style>
    .main {
        background-image: linear-gradient(to bottom right, #38ebd95f, #9d38eb5f);
    }

    .stFileUploader > label div[role='button'] {
        background-color: #1f77b4;
        color: white;
        padding: 0.5em 1em;
        border-radius: 6px;
        border: none;
        text-align: center;
        transition: background-color 0.3s ease;
    }
   
    .stButton button {
        background-color: #1f78b4;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
    }

    .stButton button:hover {
        background-color:  white;
        color: #3bd440;
        border-color: #3bd440;
        cursor: pointer
    }

    .stTextArea textarea {
        background-color: #eaf1f8;
        color: #333;
    }
    .text {
        color: black;
    }
    </style>
""", unsafe_allow_html=True)


st.title("Speaker Diarization and Transcription")
st.markdown("Upload an audio file and click the button to generate a transcript.")

audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# AHC clustering parameters
with st.expander("Advanced Settings"):
    threshold = st.slider("AHC Clustering Threshold", 0.1, 0.5, 0.3, 0.05, 
                         help="Lower values create more speakers, higher values create fewer speakers")
    
    linkage = st.selectbox("Linkage Method", 
                          ["average", "complete", "single", "ward"], 
                          index=0,
                          help="Method to compute distance between clusters")

# Initialize session state
if 'transcript_generated' not in st.session_state:
    st.session_state.transcript_generated = False
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None
if 'speaker_segments' not in st.session_state:
    st.session_state.speaker_segments = {}
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'num_speakers' not in st.session_state:
    st.session_state.num_speakers = 0
if 'speaker_names' not in st.session_state:
    st.session_state.speaker_names = {}

def update_transcript_with_names():
    """Update transcript with custom speaker names"""
    if not st.session_state.speaker_segments:
        return st.session_state.transcript
    
    # Create a new transcript with custom names
    all_segments = []
    for speaker_id, segments in st.session_state.speaker_segments.items():
        for segment in segments:
            if "transcription" in segment and segment["transcription"].strip():
                all_segments.append({
                    "speaker_id": speaker_id,
                    "start": segment["start"],
                    "transcription": segment["transcription"]
                })
    
    # Sort segments by start time
    all_segments.sort(key=lambda x: x["start"])
    
    # Generate transcript with custom names
    return_value = []
    for segment in all_segments:
        speaker_id = segment['speaker_id']
        # Use custom name if available, otherwise use default Speaker X
        if speaker_id in st.session_state.speaker_names and st.session_state.speaker_names[speaker_id]:
            speaker_name = st.session_state.speaker_names[speaker_id]
        else:
            speaker_name = f"Speaker {int(speaker_id) + 1}"
        
        return_value.append(f"[{speaker_name}]: {segment['transcription']}")
        return_value.append("\n")
    
    return ''.join(return_value)

if audio and st.button("Generate Transcript"):
    with st.spinner("Running speaker diarization..."):
        # Create directory for files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_dir = os.path.join(base_dir, "sample_audio")
        ensure_directory_exists(sample_dir)
        
        # Create unique session folder
        output_dir = os.path.join(sample_dir, f"session_{timestamp}")
        ensure_directory_exists(output_dir)
        
        # Save uploaded file
        file_extension = os.path.splitext(audio.name)[1]
        input_filename = f"input_audio{file_extension}"
        audio_path = os.path.join(output_dir, input_filename)
        
        with open(audio_path, "wb") as f:
            f.write(audio.getbuffer())
        
        st.info(f"Saved input audio to: {output_dir}")
        
        # Run the pipeline
        try:
            # Step 1: Noise Removal
            st.text("Removing background noise...")
            denoised_file = os.path.join(output_dir, "denoised.wav")
            cleaned_audio = noise_removal(audio_path, denoised_file, False)
            
            # Step 2: Audio Segmentation
            st.text("Segmenting audio...")
            segment_dir = segment_audio(cleaned_audio, output_dir, False)
            
            # Step 3: Feature Extraction (d-vectors)
            st.text("Extracting d-vector speaker features...")
            embeddings = extract_features(segment_dir, output_dir, False)
            
            # Step 4: Speaker Clustering (AHC)
            st.text("Clustering speakers using AHC...")
            diarization_result = cluster_speakers(
                embeddings, 
                output_dir, 
                None,  # Auto-determine number of speakers 
                threshold, 
                linkage, 
                False
            )
            
            # Step 5: Transcription
            st.text("Transcribing audio...")
            final_result = transcribe_audio(segment_dir, diarization_result, output_dir, "medium")
            
            # Load the full result
            result_path = os.path.join(output_dir, "full_result.json")
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            
            # Store in session state
            st.session_state.speaker_segments = result_data["speaker_segments"]
            st.session_state.num_speakers = result_data["num_speakers"]
            st.session_state.output_dir = output_dir
            
            # Initialize speaker_names
            st.session_state.speaker_names = {speaker_id: "" for speaker_id in st.session_state.speaker_segments.keys()}
            
            # Combine all segments
            all_segments = []
            for speaker_id, segments in result_data["speaker_segments"].items():
                for segment in segments:
                    if "transcription" in segment and segment["transcription"].strip():
                        all_segments.append({
                            "speaker_id": speaker_id,
                            "start": segment["start"],
                            "transcription": segment["transcription"]
                        })

            # Sort by time
            all_segments.sort(key=lambda x: x["start"])

            # Generate transcript
            return_value = []
            for segment in all_segments:
                speaker_name = f"Speaker {int(segment['speaker_id']) + 1}"
                return_value.append(f"[{speaker_name}]: {segment['transcription']}")
                return_value.append("\n")

            if not return_value:
                return_value = ["No transcribable speech detected in the audio."]
                st.session_state.transcript = "No transcribable speech detected in the audio."
            else:
                st.session_state.transcript = ''.join(return_value)
            
            st.session_state.transcript_generated = True
        
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            st.session_state.transcript = f"Error: {str(e)}"
            st.session_state.transcript_generated = False

# Display transcript and name assignment form if transcript has been generated
if st.session_state.transcript_generated:
    st.subheader("Transcript")
    st.text_area("Diarized Transcript", st.session_state.transcript, height=300)
    
    # Display form for assigning speaker names
    st.subheader("Assign Speaker Names")
    st.markdown("Assign custom names to speakers in the transcript:")
    
    # Create input fields for each speaker
    name_inputs_changed = False
    with st.form("speaker_names_form"):
        for speaker_id in sorted(st.session_state.speaker_segments.keys(), key=lambda x: int(x)):
            default_name = f"Speaker {int(speaker_id) + 1}"
            speaker_name = st.text_input(
                f"Name for {default_name}:", 
                key=f"speaker_{speaker_id}",
                value=st.session_state.speaker_names.get(speaker_id, "")
            )
            st.session_state.speaker_names[speaker_id] = speaker_name
        
        # Submit button
        submit_names = st.form_submit_button("Update Transcript with Names")
    
    if submit_names:
        # Update transcript with custom names
        updated_transcript = update_transcript_with_names()
        
        # Display updated transcript
        st.subheader("Updated Transcript with Custom Names")
        st.text_area("Transcript with Names", updated_transcript, height=300)
        
        # Save updated transcript
        if st.session_state.output_dir:
            named_transcript_path = os.path.join(st.session_state.output_dir, "transcript_with_names.txt")
            try:
                with open(named_transcript_path, 'w') as transcript_file:
                    transcript_file.write(updated_transcript)
                st.success(f"Updated transcript saved to: {named_transcript_path}")
            except Exception as e:
                st.warning(f"Could not save updated transcript file: {str(e)}")
    
    # Save original transcript if not already saved
    if st.session_state.output_dir:
        transcript_path = os.path.join(st.session_state.output_dir, "transcript.txt")
        try:
            if not os.path.exists(transcript_path):
                with open(transcript_path, 'w') as transcript_file:
                    transcript_file.write(st.session_state.transcript)
            st.success(f"All processing files and transcript saved to: {st.session_state.output_dir}")
        except Exception as e:
            st.warning(f"Could not save transcript file: {str(e)}")


