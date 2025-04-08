import os
import numpy as np
import torch
import librosa
from tqdm import tqdm
from resemblyzer import VoiceEncoder
from pathlib import Path
import matplotlib.pyplot as plt

class FeatureExtractor:
    """
    Extract d-vector speaker embeddings from audio segments using Resemblyzer.
    """
    
    def __init__(self, device=None, overlap_ratio=0.5):
        """
        Initialize the d-vector feature extractor.
        
        Args:
            device: Device to run the model on ("cpu", "cuda", "mps" for Apple Silicon)
            overlap_ratio: Overlap ratio between frames for sliding window analysis
        """
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
        
        # Initialize d-vector encoder
        self.encoder = VoiceEncoder(device=self.device)
        self.embedding_dim = 256  # d-vector dimension from Resemblyzer
    
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
            return self.extract_dvector(audio, sr)
        
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
                emb = self.extract_dvector(window, sr)
                embeddings.append(emb)
                voice_activity.append(1.0)
            else:
                voice_activity.append(0.0)
        
        # If no speech detected, just use the whole segment
        if len(embeddings) == 0:
            return self.extract_dvector(audio, sr)
        
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
        Process all audio segments in a directory and extract d-vector embeddings
        
        Args:
            segment_dir: Directory containing audio segments
            use_sliding_window: Whether to use sliding window approach
            
        Returns:
            Dictionary with segment filenames and their embeddings
        """
        segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.wav')])
        embeddings = {}
        
        print(f"Extracting d-vector embeddings for {len(segment_files)} segments...")
        for segment_file in tqdm(segment_files):
            segment_path = os.path.join(segment_dir, segment_file)
            audio, sr = librosa.load(segment_path, sr=None)
            
            # Extract embedding with or without sliding window
            if use_sliding_window:
                embedding = self.extract_with_sliding_window(audio, sr)
            else:
                embedding = self.extract_dvector(audio, sr)
            
            # Store with filename as key
            embeddings[segment_file] = embedding
        
        return embeddings
    
    def visualize_embeddings(self, embeddings, output_path=None):
        """
        Visualize embeddings using PCA
        
        Args:
            embeddings: Dictionary of embeddings
            output_path: Path to save the visualization
        """
        from sklearn.decomposition import PCA
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
        
        plt.title('PCA visualization of d-vector embeddings')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    segment_dir = "../audio_files/segments"
    
    # Create feature extractor for d-vectors
    extractor = FeatureExtractor(overlap_ratio=0.5)
    
    # Process segments and extract embeddings with sliding window
    embeddings = extractor.process_segments(segment_dir, use_sliding_window=True)
    
    # Save embeddings
    np.save("../audio_files/segment_embeddings.npy", embeddings)
    
    # Visualize embeddings
    extractor.visualize_embeddings(embeddings, "../audio_files/embedding_visualization.png")
    
    print(f"Extracted {len(embeddings)} embeddings with dimension {next(iter(embeddings.values())).shape}")