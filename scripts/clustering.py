import os
import numpy as np
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hclust
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
from sklearn.metrics import silhouette_score


class SpeakerClustering:
    """
    Agglomerative Hierarchical Clustering for speaker diarization
    """
    
    def __init__(self):
        self.labels_ = None
        self.embeddings = None
        self.segment_names = None
    
    def _prepare_embeddings(self, embeddings_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        segment_names = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[name] for name in segment_names])
        
        # Normalize embeddings to unit length for cosine similarity
        embeddings = normalize(embeddings)
        
        return embeddings, segment_names
    
    def _detect_outliers(self, embeddings: np.ndarray, threshold: float = 20000000000.0) -> np.ndarray:
        # Compute the mean and standard deviation of the embeddings
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)

        # Compute Z-scores for each embedding
        z_scores = np.abs((embeddings - mean) / std)

        # Identify outliers based on the threshold
        outlier_mask = np.any(z_scores > threshold, axis=1)
        return outlier_mask

    def perform_clustering(self, embeddings_dict: Dict[str, np.ndarray], 
                          num_speakers: int = None, threshold: float = 0.3,
                          linkage: str = 'average', outlier_threshold: float = 3.0):
        # Prepare embeddings
        X, segment_names = self._prepare_embeddings(embeddings_dict)
        self.embeddings = X
        self.segment_names = segment_names

        print(f"Clustering with AHC using {linkage} linkage")

        # Step 1: Detect outliers
        # outlier_mask = self._detect_outliers(X, threshold=outlier_threshold)
        outlier_mask = np.zeros(X.shape[0], dtype=bool)
        non_outliers = ~outlier_mask
        print(f"Detected {np.sum(outlier_mask)} outliers out of {len(X)} segments")

        # Separate outliers and non-outliers
        X_non_outliers = X[non_outliers]
        segment_names_non_outliers = [segment_names[i] for i in range(len(segment_names)) if non_outliers[i]]
        outlier_indices = np.where(outlier_mask)[0]

        # Handle case with too few non-outlier segments
        if len(X_non_outliers) < 2:
            print("Too few non-outlier segments for clustering, defaulting to 1 speaker")
            self.labels_ = np.zeros(len(X), dtype=int)
            result = {segment_names[i]: 0 for i in range(len(segment_names))}
            return result

        # Step 2: Calculate distance matrix once (for non-outliers)
        distance_matrix = dist.squareform(dist.pdist(X_non_outliers, metric='cosine'))

        # Step 3: Determine optimal number of clusters if not provided
        if num_speakers is None:
            best_k = None
            best_sil = -1.0
            min_k = 2  # At least 2 speakers
            max_k = min(8, len(X_non_outliers) - 1)  # At most 8 speakers or n-1 

            if max_k >= min_k:
                print(f"Trying {min_k}-{max_k} clusters with silhouette scoring for non-outliers")

                for k in range(min_k, max_k + 1):
                    temp_clustering = AgglomerativeClustering(
                        n_clusters=k,
                        metric='precomputed' if linkage != 'ward' else 'euclidean',
                        linkage=linkage
                    )
                    temp_labels = temp_clustering.fit_predict(
                        distance_matrix if linkage != 'ward' else X_non_outliers
                    )

                    # Check if we have at least 2 samples per cluster for silhouette
                    cluster_counts = np.bincount(temp_labels)

                    # count number of clusters with fewer than 2 samples
                    if np.sum(cluster_counts < 2) > 2:
                        print(f"Skipping k={k} - too many clusters with fewer than 2 samples")
                        continue

                    score = silhouette_score(
                        X_non_outliers if linkage == 'ward' else distance_matrix, 
                        temp_labels, 
                        metric='cosine' if linkage != 'ward' else 'euclidean'
                    )
                    print(f"k={k}, silhouette={score:.3f}")

                    if score > best_sil:
                        best_sil = score
                        best_k = k

                # Step 4: Perform clustering with the optimal k or fallback to threshold
                if best_k is not None and best_sil > 0.1:
                    print(f"Best silhouette score {best_sil:.3f} found with {best_k} clusters")
                    num_speakers = best_k
                else:
                    print("Silhouette score too low, falling back to threshold-based clustering")
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=threshold,
                        metric='precomputed' if linkage != 'ward' else 'euclidean',
                        linkage=linkage
                    )
                    labels_non_outliers = clustering.fit_predict(
                        distance_matrix if linkage != 'ward' else X_non_outliers
                    )
            else:
                print("Not enough segments for silhouette scoring, falling back to threshold-based clustering")
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    metric='precomputed' if linkage != 'ward' else 'euclidean',
                    linkage=linkage
                )
                labels_non_outliers = clustering.fit_predict(
                    distance_matrix if linkage != 'ward' else X_non_outliers
                )

        # Step 5: Perform the final clustering with determined number of speakers or threshold
        if num_speakers is not None:
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                metric='precomputed' if linkage != 'ward' else 'euclidean',
                linkage=linkage
            )
            labels_non_outliers = clustering.fit_predict(
                distance_matrix if linkage != 'ward' else X_non_outliers
            )

        # Step 6: Initialize all labels and assign non-outliers
        labels = np.full(len(X), -1, dtype=int)
        labels[non_outliers] = labels_non_outliers

        # Step 7: Assign outliers to the nearest cluster
        if len(outlier_indices) > 0 and len(np.unique(labels_non_outliers)) > 0:
            print(f"Assigning {len(outlier_indices)} outliers to nearest clusters")
            for outlier_idx in outlier_indices:
                # Calculate average distance from this outlier to each cluster's points
                distances_to_clusters = []
                for cluster_id in np.unique(labels_non_outliers):
                    # Get points belonging to this cluster
                    cluster_points = X_non_outliers[labels_non_outliers == cluster_id]
                    # Calculate distances from outlier to all points in cluster
                    distances = dist.cdist([X[outlier_idx]], cluster_points, metric='cosine')[0]
                    # Use mean distance to cluster
                    distances_to_clusters.append(np.mean(distances))
                
                # Assign to closest cluster
                closest_cluster = np.unique(labels_non_outliers)[np.argmin(distances_to_clusters)]
                labels[outlier_idx] = closest_cluster

        self.labels_ = labels

        # Create result dictionary mapping segment names to speaker labels
        result = {segment_names[i]: int(self.labels_[i]) for i in range(len(segment_names))}

        # Print cluster sizes
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Speaker {label}: {count} segments")

        return result
    
    def visualize_clusters(self, output_path=None):
        if self.labels_ is None or self.embeddings is None:
            raise ValueError("No clustering results to visualize. Run perform_clustering first.")
            
        from sklearn.decomposition import PCA
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
        
        plt.title('PCA visualization of AHC clustering')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_diarization_result(self, segment_to_speaker: Dict[str, int], 
                                   segment_timing_info: Dict[str, Dict[str, float]] = None) -> Dict:
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
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Diarization results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Load embeddings
    embeddings = np.load("../audio_files/segment_embeddings.npy", allow_pickle=True).item()
    
    # Create clustering model
    clustering = SpeakerClustering()
    
    # Perform clustering
    segment_to_speaker = clustering.perform_clustering(embeddings, threshold=0.3)
    
    # Visualize clusters
    clustering.visualize_clusters("../audio_files/clustering_visualization.png")
    
    # Generate diarization result
    diarization_result = clustering.generate_diarization_result(segment_to_speaker)
    
    # Save results
    clustering.save_diarization_result(diarization_result, "../audio_files/diarization_result.json")