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
    1. MFCC (traditional features)
    2. d-vectors (using Resemblyzer's VoiceEncoder)
    3. x-vectors (using a simplified implementation)
    """
    
    def __init__(self, method="dvector", device=None):
        """
        Initialize the feature extractor.
        
        Args:
            method: Feature extraction method - "mfcc", "dvector", or "xvector"
            device: Device to run the model on ("cpu", "cuda", "mps" for Apple Silicon)
        """
        self.method = method
        
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
        elif method == "xvector":
            # Load a pretrained x-vector model or initialize a simpler one
            self.init_xvector_model()
            self.embedding_dim = 512  # typical x-vector dimension
    
    def init_xvector_model(self):
        """Initialize a simplified x-vector model"""
        # This is a simplified TDNN-based architecture inspired by x-vector
        # In a real implementation, you would load a pre-trained model or train one
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(40, 128, kernel_size=5, dilation=1),  # Frame-level processing
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=3, dilation=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=3, dilation=3),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
        ).to(self.device)
        
        # These layers would be applied after pooling for segment-level processing
        self.segment_layers = torch.nn.Sequential(
            torch.nn.Linear(256, 512),  # 256 = 128*2 from stats pooling
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 512),  # This output is the x-vector
        ).to(self.device)
    
    def extract_mfcc(self, audio, sr):
        """Extract MFCC features from audio"""
        # Extract MFCCs using librosa
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        
        # Return mean and std as a simple embedding
        return np.hstack([np.mean(features, axis=1), np.std(features, axis=1)])
    
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
    
    def extract_xvector(self, audio, sr):
        """Extract x-vector embedding using a simplified model"""
        # Ensure 16kHz sample rate for consistency
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Extract filter bank features (40 dimensions)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        log_mel = librosa.power_to_db(mel_spec)
        
        # Normalize features
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        
        # Convert to PyTorch tensor and reshape for 1D convolution
        features = torch.FloatTensor(log_mel).unsqueeze(0).to(self.device)  # [1, 40, time]
        
        # Forward pass through frame-level layers
        with torch.no_grad():
            frame_embeddings = self.model(features)
            
            # Simple stats pooling (mean and std)
            mean = torch.mean(frame_embeddings, dim=2)
            std = torch.std(frame_embeddings, dim=2)
            pooled = torch.cat([mean, std], dim=1)
            
            # Forward pass through segment-level layers
            embedding = self.segment_layers(pooled)
        
        return embedding.cpu().numpy().flatten()
    
    def extract_embedding(self, audio, sr):
        """Extract embedding based on the selected method"""
        if self.method == "mfcc":
            return self.extract_mfcc(audio, sr)
        elif self.method == "dvector":
            return self.extract_dvector(audio, sr)
        elif self.method == "xvector":
            return self.extract_xvector(audio, sr)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def process_segments(self, segment_dir):
        """
        Process all audio segments in a directory and extract embeddings
        
        Args:
            segment_dir: Directory containing audio segments
            
        Returns:
            Dictionary with segment filenames and their embeddings
        """
        segment_files = sorted([f for f in os.listdir(segment_dir) if f.endswith('.wav')])
        embeddings = {}
        
        print(f"Extracting {self.method} embeddings for {len(segment_files)} segments...")
        for segment_file in tqdm(segment_files):
            segment_path = os.path.join(segment_dir, segment_file)
            audio, sr = librosa.load(segment_path, sr=None)
            
            # Extract embedding
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
    
    # Create feature extractor with d-vector embeddings
    extractor = FeatureExtractor(method="dvector")
    
    # Process segments and extract embeddings
    embeddings = extractor.process_segments(segment_dir)
    
    # Save embeddings
    np.save("../audio_files/segment_embeddings.npy", embeddings)
    
    # Visualize embeddings
    extractor.visualize_embeddings(embeddings, "../audio_files/embedding_visualization.png")
    
    print(f"Extracted {len(embeddings)} embeddings with dimension {next(iter(embeddings.values())).shape}")