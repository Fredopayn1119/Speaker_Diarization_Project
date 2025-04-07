import os
import numpy as np
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hclust
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any


class SpeakerClustering:
    """
    Clustering methods for speaker diarization.
    Supports different clustering algorithms:
    - Agglomerative Hierarchical Clustering (AHC) - Can automatically determine speakers with threshold
    - DBSCAN - Automatically determines number of clusters based on density
    
    Methods requiring pre-specified speaker count (now removed):
    - Spectral Clustering
    - K-Means
    """
    
    def __init__(self, method="ahc"):
        """
        Initialize the clustering model.
        
        Args:
            method: Clustering method - "ahc" or "dbscan"
                   (methods that can automatically determine number of speakers)
        """
        if method.lower() not in ["ahc", "dbscan"]:
            print(f"WARNING: Method '{method}' requires pre-specifying number of speakers. Defaulting to 'ahc'.")
            method = "ahc"
            
        self.method = method.lower()
        self.labels_ = None
        self.embeddings = None
        self.segment_names = None
    
    def _prepare_embeddings(self, embeddings_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert embeddings dictionary to matrix and normalize.
        
        Args:
            embeddings_dict: Dictionary mapping segment names to embeddings
            
        Returns:
            Tuple of (embeddings_matrix, segment_names)
        """
        segment_names = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[name] for name in segment_names])
        
        # Normalize embeddings to unit length for cosine similarity
        embeddings = normalize(embeddings)
        
        return embeddings, segment_names
    
    def estimate_num_speakers(self, embeddings: np.ndarray, max_speakers: int = 10) -> int:
        """
        Estimate the number of speakers using various heuristics.
        
        Args:
            embeddings: Matrix of speaker embeddings
            max_speakers: Maximum number of speakers to consider
            
        Returns:
            Estimated number of speakers
        
        Raises:
            ValueError: If unable to determine the number of speakers
        """
        # Calculate distance matrix (using cosine distance)
        distance_matrix = dist.squareform(dist.pdist(embeddings, metric='cosine'))
        
        # Method 1: Use linkage and inconsistency to estimate optimal clusters
        Z = hclust.linkage(distance_matrix, method='average')
        
        # Try different thresholds and evaluate silhouette score
        best_score = -1
        best_k = -1  # Default to -1 to indicate "not found"
        
        for k in range(2, min(max_speakers + 1, len(embeddings))):
            # Skip if we have too few samples compared to clusters
            if k >= len(embeddings) - 1:
                continue
                
            # AHC with k clusters - use parameters compatible with all scikit-learn versions
            try:
                # First try with affinity parameter (newer scikit-learn versions)
                clustering = AgglomerativeClustering(
                    n_clusters=k, 
                    affinity='precomputed', 
                    linkage='average'
                )
                labels = clustering.fit_predict(distance_matrix)
            except TypeError:
                # Fall back to older scikit-learn versions or different parameter format
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=k,
                        linkage='average'
                    )
                    labels = clustering.fit_predict(distance_matrix)
                except:
                    # If both approaches fail, use a very basic approach
                    print("Warning: Advanced clustering methods failed, using basic approach")
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score (higher is better)
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(embeddings, labels, metric='cosine')
                
                print(f"  {k} clusters: silhouette = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # Method 2: Eigengap heuristic for spectral clustering (simpler version)
        eigengap_k = -1  # Default to -1 to indicate "not found"
        
        # if len(embeddings) > 3:  # Only if we have enough samples
        #     # Create affinity matrix from distance matrix
        #     affinity_matrix = 1 - distance_matrix
            
        #     # Set small values to zero (sparse affinity)
        #     threshold = np.percentile(affinity_matrix.flatten(), 50)  # Median as threshold
        #     affinity_matrix[affinity_matrix < threshold] = 0
            
        #     # Calculate Laplacian
        #     from scipy.sparse.linalg import eigsh
        #     from scipy.sparse import csr_matrix
            
        #     D = np.diag(np.sum(affinity_matrix, axis=1))
        #     L = D - affinity_matrix
        #     L_sparse = csr_matrix(L)
            
        #     # Get eigenvalues
        #     try:
        #         eigenvalues, _ = eigsh(L_sparse, k=min(max_speakers+5, len(embeddings)-1), 
        #                               which='SM', return_eigenvectors=True)
                
        #         # Sort eigenvalues
        #         eigenvalues = sorted(eigenvalues)
                
        #         # Find largest eigengap
        #         eigengaps = np.diff(eigenvalues)
        #         if len(eigengaps) > 1:
        #             # Add 1 because gap is between eigenvalues
        #             eigengap_k = np.argmax(eigengaps[:max_speakers-1]) + 1
                    
        #             print(f"Eigengap heuristic suggests {eigengap_k} speakers")
        #     except:
        #         print("Eigengap calculation failed, using silhouette score only")
        
        # Combine results from both methods
        if best_k != -1 and eigengap_k != -1:
            # If both methods found a result, average them
            final_k = (best_k + eigengap_k) // 2
            print(f"Combined estimate: {final_k} speakers (silhouette: {best_k}, eigengap: {eigengap_k})")
        elif best_k != -1:
            # Only silhouette method worked
            final_k = best_k
            print(f"Estimated number of speakers: {final_k} (based on silhouette)")
        elif eigengap_k != -1:
            # Only eigengap method worked
            final_k = eigengap_k
            print(f"Estimated number of speakers: {final_k} (based on eigengap)")
        else:
            # No method worked
            raise ValueError("Unable to determine the number of speakers. Please specify the number of speakers manually.")
        
        return final_k
    
    def perform_clustering(self, embeddings_dict: Dict[str, np.ndarray], 
                          num_speakers: int = None, threshold: float = 0.3):
        """
        Perform clustering on speaker embeddings.
        
        Args:
            embeddings_dict: Dictionary mapping segment names to embeddings
            num_speakers: Number of speakers (estimated automatically if None)
            threshold: Distance threshold for AHC (default 0.3, determines clusters automatically)
            
        Returns:
            Dictionary mapping segment names to speaker labels
        """
        # Prepare embeddings
        X, segment_names = self._prepare_embeddings(embeddings_dict)
        self.embeddings = X
        self.segment_names = segment_names
        
        # Estimate number of speakers if not provided (just for information)
        estimated_speakers = None
        if num_speakers is None:
            try:
                print("Estimating number of speakers (for reference)...")
                estimated_speakers = self.estimate_num_speakers(X)
                print(f"Estimated {estimated_speakers} speakers, but will use automatic clustering")
            except ValueError as e:
                print(f"Could not estimate speakers: {e}")
                print("Will use automatic clustering methods")
        
        print(f"Clustering with method: {self.method}")
        
        if self.method == "ahc":
            # Agglomerative Hierarchical Clustering with automatic threshold
            
            # Calculate distance matrix (cosine distance)
            distance_matrix = dist.squareform(dist.pdist(X, metric='cosine'))
            
            try:
                # Try with affinity parameter (newer scikit-learn)
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    affinity='precomputed',
                    linkage='average'
                )
                self.labels_ = clustering.fit_predict(distance_matrix)
            except TypeError:
                # Fall back to older versions or different parameter formats
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=threshold,
                        linkage='average'
                    )
                    self.labels_ = clustering.fit_predict(distance_matrix)
                except Exception as e:
                    print(f"Error with threshold-based clustering: {e}")
                    # If we have an estimate, use that as a fallback
                    if estimated_speakers is not None:
                        print(f"Using estimated speakers ({estimated_speakers}) as fallback")
                        num_speakers = estimated_speakers
                    else:
                        # Default to 2 speakers if we can't estimate
                        print("Default to 2 speakers as fallback")
                        num_speakers = 2
                        
                    clustering = AgglomerativeClustering(
                        n_clusters=num_speakers,
                        linkage='average'
                    )
                    self.labels_ = clustering.fit_predict(X)
            
            print(f"AHC with threshold {threshold} found {len(np.unique(self.labels_))} speakers")
            
        elif self.method == "dbscan":
            # DBSCAN (automatically determines number of clusters)
            
            # Heuristic to determine eps: use mean of distances to k-nearest neighbors
            from sklearn.neighbors import NearestNeighbors
            
            # Use estimated speakers if available, otherwise default to 2
            k_speakers = estimated_speakers if estimated_speakers is not None else 2
            k = min(len(X) - 1, max(k_speakers * 2, 5))
            
            nbrs = NearestNeighbors(n_neighbors=k).fit(X)
            distances, _ = nbrs.kneighbors(X)
            
            # Use average of k-distances as eps
            eps = np.mean(distances[:, -1]) * 0.5  # Adjust this multiplier if needed
            
            clustering = DBSCAN(
                eps=eps,
                min_samples=max(2, len(X) // 10),  # More reasonable default
                metric='cosine'
            )
            self.labels_ = clustering.fit_predict(X)
            
            # Handle noise points (-1 label) by assigning to nearest cluster
            if -1 in self.labels_:
                noise_indices = np.where(self.labels_ == -1)[0]
                if len(noise_indices) > 0:
                    print(f"Found {len(noise_indices)} noise points, reassigning to nearest cluster")
                    
                    # Get non-noise cluster centers
                    valid_labels = np.unique(self.labels_[self.labels_ >= 0])
                    
                    # If no valid clusters were found, create a single cluster for all points
                    if len(valid_labels) == 0:
                        print("No valid clusters found, creating a single cluster")
                        self.labels_[:] = 0
                    else:
                        centers = np.array([X[self.labels_ == l].mean(axis=0) for l in valid_labels])
                        
                        # Assign noise points to nearest center
                        for idx in noise_indices:
                            dists = [dist.cosine(X[idx], center) for center in centers]
                            closest = valid_labels[np.argmin(dists)]
                            self.labels_[idx] = closest
        
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")
        
        # Create result dictionary mapping segment names to speaker labels
        result = {segment_names[i]: int(self.labels_[i]) for i in range(len(segment_names))}
        
        # Print cluster sizes
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Speaker {label}: {count} segments")
        
        return result
    
    def visualize_clusters(self, output_path=None):
        """
        Visualize clustering results using PCA or t-SNE
        
        Args:
            output_path: Path to save the visualization
        """
        if self.labels_ is None or self.embeddings is None:
            raise ValueError("No clustering results to visualize. Run perform_clustering first.")
            
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.embeddings)
        
        # Visualize PCA with cluster colors
        plt.figure(figsize=(12, 10))
        
        # Get a color map with enough colors
        cmap = plt.get_cmap('tab10')
        unique_labels = np.unique(self.labels_)
        
        for label in unique_labels:
            mask = self.labels_ == label
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       label=f'Speaker {label}',
                       color=cmap(label % 10),
                       alpha=0.8)
        
        # Add some segment names as labels
        for i, name in enumerate(self.segment_names):
            if i % max(1, len(self.segment_names) // 10) == 0:  # Label some points
                plt.annotate(name, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
        
        plt.title(f'PCA visualization of {self.method.upper()} clustering')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # t-SNE visualization if enough samples are available
        if len(self.segment_names) >= 10:
            # Ensure perplexity is less than n_samples
            perplexity_value = min(30, len(self.embeddings) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=250)
            X_tsne = tsne.fit_transform(self.embeddings)
            
            plt.figure(figsize=(12, 10))
            for label in unique_labels:
                mask = self.labels_ == label
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           label=f'Speaker {label}',
                           color=cmap(label % 10),
                           alpha=0.8)
            
            plt.title(f't-SNE visualization of {self.method.upper()} clustering')
            plt.legend()
            
            if output_path:
                tsne_path = output_path.replace('.png', '_tsne.png')
                plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
    
    def generate_diarization_result(self, segment_to_speaker: Dict[str, int], 
                                   segment_timing_info: Dict[str, Dict[str, float]] = None) -> Dict:
        """
        Generate structured diarization results from clustering.
        
        Args:
            segment_to_speaker: Dictionary mapping segment names to speaker IDs
            segment_timing_info: Dictionary with segment timing information
                                (if None, will try to parse from filenames)
            
        Returns:
            Dictionary with structured diarization results
        """
        # Initialize result structure
        result = {
            "num_speakers": len(set(segment_to_speaker.values())),
            "speaker_segments": {}
        }
        
        # Initialize speaker segments
        for speaker_id in set(segment_to_speaker.values()):
            result["speaker_segments"][str(speaker_id)] = []
        
        # Process each segment
        for segment_name, speaker_id in segment_to_speaker.items():
            # If timing info is provided, use it
            if segment_timing_info and segment_name in segment_timing_info:
                start = segment_timing_info[segment_name]["start"]
                end = segment_timing_info[segment_name]["end"]
            else:
                # Try to parse timing from filename (e.g., segment_10.wav)
                # Assuming segments are of fixed length (e.g., 1 second)
                try:
                    # Extract segment number
                    import re
                    match = re.search(r'segment_(\d+)', segment_name)
                    if match:
                        segment_num = int(match.group(1))
                        # Assuming 1-second segments
                        segment_length = 1.0
                        start = (segment_num - 1) * segment_length
                        end = segment_num * segment_length
                    else:
                        # If can't parse, use index as time
                        start = 0
                        end = 1
                except:
                    # Fallback
                    start = 0
                    end = 1
            
            # Add segment to appropriate speaker
            result["speaker_segments"][str(speaker_id)].append({
                "segment": segment_name,
                "start": start,
                "end": end
            })
        
        # Sort segments by start time for each speaker
        for speaker_id in result["speaker_segments"]:
            result["speaker_segments"][speaker_id].sort(key=lambda x: x["start"])
            
            # Merge adjacent segments from the same speaker
            merged_segments = []
            current_segment = None
            
            for segment in result["speaker_segments"][speaker_id]:
                if current_segment is None:
                    current_segment = segment.copy()
                elif abs(segment["start"] - current_segment["end"]) < 0.1:
                    # Merge if segments are close
                    current_segment["end"] = segment["end"]
                    current_segment["segment"] += "+" + segment["segment"].split("/")[-1]
                else:
                    # Add completed segment and start a new one
                    merged_segments.append(current_segment)
                    current_segment = segment.copy()
            
            # Add the last segment
            if current_segment:
                merged_segments.append(current_segment)
                
            result["speaker_segments"][speaker_id] = merged_segments
        
        return result
    
    def save_diarization_result(self, result: Dict, output_path: str):
        """
        Save diarization results to a JSON file.
        
        Args:
            result: Diarization result dictionary
            output_path: Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Diarization results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Load embeddings
    embeddings = np.load("../audio_files/segment_embeddings.npy", allow_pickle=True).item()
    
    # Create clustering model
    clustering = SpeakerClustering(method="ahc")
    
    # Perform clustering
    segment_to_speaker = clustering.perform_clustering(embeddings, num_speakers=None)
    
    # Visualize clusters
    clustering.visualize_clusters("../audio_files/clustering_visualization.png")
    
    # Generate diarization result
    diarization_result = clustering.generate_diarization_result(segment_to_speaker)
    
    # Save results
    clustering.save_diarization_result(diarization_result, "../audio_files/diarization_result.json")