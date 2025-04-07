import os
import numpy as np
import torch
import torchaudio
import librosa
from tqdm import tqdm
from resemblyzer import VoiceEncoder
from pathlib import Path
import matplotlib.pyplot as plt

class FeatureExtractor:
    """
    Extract speaker embeddings from audio segments using different methods:
    1. d-vectors (using Resemblyzer's VoiceEncoder)
    2. ECAPA-TDNN embeddings (state-of-the-art speaker embeddings)
    3. Wav2Vec2 embeddings (self-supervised speech representation)
    """
    
    def __init__(self, method="ecapa", device=None, overlap_ratio=0.5):
        """
        Initialize the feature extractor.
        
        Args:
            method: Feature extraction method - "dvector", "ecapa", "wav2vec"
            device: Device to run the model on ("cpu", "cuda", "mps" for Apple Silicon)
            overlap_ratio: Overlap ratio between frames for sliding window analysis
        """
        self.method = method
        self.overlap_ratio = overlap_ratio
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon support
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize models based on method
        if method == "dvector":
            self.encoder = VoiceEncoder(device=self.device)
            self.embedding_dim = 256  # d-vector dimension from Resemblyzer
        
        elif method == "ecapa":
            # Load ECAPA-TDNN pretrained model
            try:
                print("Loading ECAPA-TDNN model...")
                self.encoder = torch.hub.load('speechbrain/speechbrain', 
                                            'EncoderClassifier.from_hparams', 
                                            source='github',
                                            hparams_file='speechbrain/pretrained/embeddings/spkrec-ecapa-voxceleb/hyperparams.yaml',
                                            savedir='pretrained_models/spkrec-ecapa-voxceleb')
                self.encoder.to(self.device)
                self.embedding_dim = 192  # ECAPA-TDNN embedding dimension
                print("ECAPA-TDNN model loaded successfully")
            except Exception as e:
                print(f"Failed to load ECAPA-TDNN model: {e}")
                print("Falling back to d-vector extraction method")
                self.method = "dvector"
                self.encoder = VoiceEncoder(device=self.device)
                self.embedding_dim = 256
        
        elif method == "wav2vec":
            # Load Wav2Vec2 pretrained model
            try:
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
                
                print("Loading Wav2Vec2 model...")
                model_name = "facebook/wav2vec2-base-960h"
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.encoder = Wav2Vec2Model.from_pretrained(model_name)
                self.encoder.to(self.device)
                self.embedding_dim = 768  # Wav2Vec2 base model embedding dimension
                print("Wav2Vec2 model loaded successfully")
            except Exception as e:
                print(f"Failed to load Wav2Vec2 model: {e}")
                print("Falling back to d-vector extraction method")
                self.method = "dvector"
                self.encoder = VoiceEncoder(device=self.device)
                self.embedding_dim = 256
        else:
            # Default to d-vector if method is not recognized
            print(f"Method {method} not recognized, using d-vector instead")
            self.encoder = VoiceEncoder(device=self.device)
            self.embedding_dim = 256
            self.method = "dvector"
    
    def extract_dvector(self, audio, sr):
        """Extract d-vector embedding using Resemblyzer"""
        # Resemblyzer requires 16kHz audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Convert to float and normalize if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        # Extract the embedding (d-vector)
        embedding = self.encoder.embed_utterance(audio)
        return embedding
    
    def extract_ecapa(self, audio, sr):
        """Extract ECAPA-TDNN embeddings"""
        # ECAPA-TDNN requires 16kHz audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Convert to float and normalize
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            
        # Convert to tensor
        with torch.no_grad():
            waveform = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            embeddings = self.encoder.encode_batch(waveform)
            embedding = embeddings.squeeze(0).cpu().numpy()
            
        return embedding
    
    def extract_wav2vec(self, audio, sr):
        """Extract Wav2Vec2 embeddings"""
        # Wav2Vec2 requires 16kHz audio
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        # Convert to float and normalize
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            
        # Process audio with Wav2Vec2
        with torch.no_grad():
            input_values = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
            outputs = self.encoder(input_values)
            
            # Average hidden states across time dimension
            hidden_states = outputs.last_hidden_state.squeeze(0)
            embedding = torch.mean(hidden_states, dim=0).cpu().numpy()
            
        return embedding
        
    def extract_embedding(self, audio, sr):
        """Extract embedding based on the selected method"""
        if self.method == "dvector":
            return self.extract_dvector(audio, sr)
        elif self.method == "ecapa":
            return self.extract_ecapa(audio, sr)
        elif self.method == "wav2vec":
            return self.extract_wav2vec(audio, sr)
        else:
            # Default to d-vector
            return self.extract_dvector(audio, sr)
    
    def extract_with_sliding_window(self, audio, sr, min_segment_length=1.0):
        """
        Extract embeddings using sliding window approach for more robust representations
        
        Args:
            audio: Audio signal
            sr: Sample rate
            min_segment_length: Minimum segment length in seconds
            
        Returns:
            Aggregated embedding vector
        """
        # Skip short segments
        if len(audio) / sr < min_segment_length:
            return self.extract_embedding(audio, sr)
        
        # Create sliding windows
        window_samples = int(min_segment_length * sr)
        hop_samples = int(window_samples * (1 - self.overlap_ratio))
        
        embeddings = []
        voice_activity = []  # Track VAD for each window
        
        # Process each window
        for start in range(0, len(audio) - window_samples, hop_samples):
            window = audio[start:start + window_samples]
            
            # Simple energy-based VAD
            energy = np.mean(window**2)
            is_speech = energy > 0.0001  # Simple threshold
            
            if is_speech:
                emb = self.extract_embedding(window, sr)
                embeddings.append(emb)
                voice_activity.append(1.0)
            else:
                voice_activity.append(0.0)
        
        # If no speech detected, just use the whole segment
        if len(embeddings) == 0:
            return self.extract_embedding(audio, sr)
        
        # Weighted average of embeddings based on voice activity
        embeddings = np.array(embeddings)
        voice_activity = np.array(voice_activity)
        
        # Normalize weights
        weights = voice_activity / np.sum(voice_activity)
        
        # Compute weighted average
        avg_embedding = np.sum(embeddings * weights[:, np.newaxis], axis=0)
        
        # Normalize the embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    def process_segments(self, segment_dir, use_sliding_window=True):
        """
        Process all audio segments in a directory and extract embeddings
        
        Args:
            segment_dir: Directory containing audio segments
            use_sliding_window: Whether to use sliding window approach
            
        Returns:
            Dictionary with segment filenames and their embeddings
        """
        segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.wav')])
        embeddings = {}
        
        print(f"Extracting {self.method} embeddings for {len(segment_files)} segments...")
        for segment_file in tqdm(segment_files):
            segment_path = os.path.join(segment_dir, segment_file)
            audio, sr = librosa.load(segment_path, sr=None)
            
            # Extract embedding with or without sliding window
            if use_sliding_window:
                embedding = self.extract_with_sliding_window(audio, sr)
            else:
                embedding = self.extract_embedding(audio, sr)
            
            # Store with filename as key
            embeddings[segment_file] = embedding
        
        return embeddings
    
    def visualize_embeddings(self, embeddings, output_path=None):
        """
        Visualize embeddings using PCA or t-SNE
        
        Args:
            embeddings: Dictionary of embeddings
            output_path: Path to save the visualization
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Convert embeddings to numpy array
        embedding_list = list(embeddings.values())
        X = np.vstack(embedding_list)
        filenames = list(embeddings.keys())
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Visualize PCA
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
        
        # Add labels for some points (not all to avoid clutter)
        for i, filename in enumerate(filenames):
            if i % max(1, len(filenames) // 20) == 0:  # Label every 20th point
                plt.annotate(filename, (X_pca[i, 0], X_pca[i, 1]))
        
        plt.title(f'PCA visualization of {self.method} embeddings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # If we have more than 10 samples, also try t-SNE
        if len(filenames) >= 10:
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.8)
            
            for i, filename in enumerate(filenames):
                if i % max(1, len(filenames) // 20) == 0:  # Label every 20th point
                    plt.annotate(filename, (X_tsne[i, 0], X_tsne[i, 1]))
            
            plt.title(f't-SNE visualization of {self.method} embeddings')
            plt.grid(True, alpha=0.3)
            
            if output_path:
                tsne_path = output_path.replace('.png', '_tsne.png')
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    # Example usage
    segment_dir = "../audio_files/segments"
    
    # Create feature extractor with ECAPA-TDNN embeddings (state-of-the-art)
    extractor = FeatureExtractor(method="ecapa", overlap_ratio=0.5)
    
    # Process segments and extract embeddings with sliding window
    embeddings = extractor.process_segments(segment_dir, use_sliding_window=True)
    
    # Save embeddings
    np.save("../audio_files/segment_embeddings.npy", embeddings)
    
    # Visualize embeddings
    extractor.visualize_embeddings(embeddings, "../audio_files/embedding_visualization.png")
    
    print(f"Extracted {len(embeddings)} embeddings with dimension {next(iter(embeddings.values())).shape}")