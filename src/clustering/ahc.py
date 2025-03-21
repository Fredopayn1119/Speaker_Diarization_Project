# This file implements Agglomerative Hierarchical Clustering (AHC) for grouping speaker embeddings.

"""Agglomerative Hierarchical Clustering for speaker diarization."""

import os
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AHC_DISTANCE_THRESHOLD, MAX_NUM_SPEAKERS, MIN_NUM_SPEAKERS
)


class AHCClustering:
    """Agglomerative Hierarchical Clustering for speaker diarization."""
    
    def __init__(self, 
                distance_threshold: float = AHC_DISTANCE_THRESHOLD,
                min_speakers: int = MIN_NUM_SPEAKERS,
                max_speakers: int = MAX_NUM_SPEAKERS,
                metric: str = "cosine"):
        """
        Initialize AHC clustering.
        
        Args:
            distance_threshold: Distance threshold for clustering
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            metric: Distance metric to use
        """
        self.distance_threshold = distance_threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.metric = metric
        
    def perform_clustering(self, 
                          embeddings: List[np.ndarray], 
                          num_speakers: Optional[int] = None) -> np.ndarray:
        """
        Perform clustering on speaker embeddings.
        
        Args:
            embeddings: List of embeddings to cluster
            num_speakers: Number of speakers (if known)
            
        Returns:
            Array of cluster labels
        """
        if len(embeddings) == 0:
            return np.array([])
            
        if len(embeddings) == 1:
            return np.array([0])
            
        # Stack embeddings into a single matrix
        X = np.vstack(embeddings)
        
        if num_speakers is not None:
            # If number of speakers is known, use it directly
            clusterer = AgglomerativeClustering(
                n_clusters=num_speakers,
                affinity=self.metric,
                linkage="average"
            )
            labels = clusterer.fit_predict(X)
        else:
            # Otherwise, use distance_threshold to determine number of clusters
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                affinity=self.metric,
                linkage="average"
            )
            labels = clusterer.fit_predict(X)
            
            # Ensure number of clusters is within bounds
            num_clusters = len(np.unique(labels))
            
            if num_clusters < self.min_speakers:
                # If too few clusters, force min_speakers
                clusterer = AgglomerativeClustering(
                    n_clusters=self.min_speakers,
                    affinity=self.metric,
                    linkage="average"
                )
                labels = clusterer.fit_predict(X)
            elif num_clusters > self.max_speakers:
                # If too many clusters, force max_speakers
                clusterer = AgglomerativeClustering(
                    n_clusters=self.max_speakers,
                    affinity=self.metric,
                    linkage="average"
                )
                labels = clusterer.fit_predict(X)
                
        return labels
    
    def find_optimal_num_speakers(self, 
                                 embeddings: List[np.ndarray], 
                                 min_clusters: int = 1, 
                                 max_clusters: int = 10) -> int:
        """
        Find the optimal number of speakers using silhouette score.
        
        Args:
            embeddings: List of embeddings
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of speakers
        """
        if len(embeddings) == 0:
            return 0
            
        if len(embeddings) == 1:
            return 1
            
        # Stack embeddings into a single matrix
        X = np.vstack(embeddings)
        
        max_clusters = min(max_clusters, len(X) - 1)
        min_clusters = min(min_clusters, max_clusters)
        
        if min_clusters == max_clusters:
            return min_clusters
            
        # Try different numbers of clusters and compute silhouette score
        silhouette_scores = []
        for n_clusters in range(min_clusters, max_clusters + 1):
            if n_clusters <= 1 or n_clusters >= len(X):
                silhouette_scores.append(-1)  # Invalid score
                continue
                
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=self.metric,
                linkage="average"
            )
            labels = clusterer.fit_predict(X)
            
            # Compute silhouette score if there are at least 2 clusters
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels, metric=self.metric)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(-1)  # Invalid score
                
        # Return the number of clusters with the highest silhouette score
        if silhouette_scores:
            best_n_clusters = min_clusters + np.argmax(silhouette_scores)
            return best_n_clusters
        else:
            return min_clusters
    
    def cluster_embeddings(self, 
                          embeddings: List[np.ndarray], 
                          segments: List[Tuple[float, float]],
                          num_speakers: Optional[int] = None) -> Dict[int, List[Tuple[float, float]]]:
        """
        Cluster embeddings and group segments by speaker.
        
        Args:
            embeddings: List of speaker embeddings
            segments: List of (start_time, end_time) tuples
            num_speakers: Number of speakers (if known)
            
        Returns:
            Dictionary mapping speaker_id to list of segments
        """
        if len(embeddings) == 0 or len(segments) == 0:
            return {}
            
        # Determine number of speakers if not provided
        if num_speakers is None:
            num_speakers = self.find_optimal_num_speakers(
                embeddings, 
                min_clusters=self.min_speakers,
                max_clusters=self.max_speakers
            )
            
        # Perform clustering
        labels = self.perform_clustering(embeddings, num_speakers)
        
        # Group segments by speaker
        speaker_segments = {}
        
        for i, (label, segment) in enumerate(zip(labels, segments)):
            if label not in speaker_segments:
                speaker_segments[label] = []
                
            speaker_segments[label].append(segment)
            
        return speaker_segments
