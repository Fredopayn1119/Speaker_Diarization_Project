#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# Import required functions and classes from existing modules
from scripts.noise_removal import noise_removal_native
from scripts.segmentation import segment_audio_native
from scripts.feature_extraction import FeatureExtractor
from scripts.clustering import SpeakerClustering

def main():
    # ----- Configuration -----
    # Specify your sample audio file here
    input_audio = "audio/New Recording 38.wav"  # <-- update with your audio file path
    output_dir = "eval"  # output directory for all results
    os.makedirs(output_dir, exist_ok=True)

    # ----- Step 1: Noise Removal -----
    denoised_audio = os.path.join(output_dir, "denoised.wav")
    print("Running noise removal...")
    noise_removal_native(input_audio, denoised_audio)

    # ----- Step 2: Segmentation -----
    segments_dir = os.path.join(output_dir, "segments")
    print("Running segmentation...")
    # Save segmentation plot to visualize detected segments
    segmentation_plot = os.path.join(output_dir, "segmentation_plot.png")
    segments = segment_audio_native(denoised_audio, segments_dir, plot_output_path=segmentation_plot, show_plot=False)

    # ----- Step 3 & 4: Feature Extraction and Clustering -----
    feature_methods = ["dvector"]
    # Updated to only include methods that automatically determine number of speakers
    clustering_methods = ["ahc"]

    # Dictionary to store embeddings for each feature extraction method
    embeddings_results = {}

    for feat_method in feature_methods:
        print(f"\nExtracting features using method: {feat_method}")
        extractor = FeatureExtractor(overlap_ratio=0.5)
        embeddings = extractor.process_segments(segments_dir, use_sliding_window=True)
        embeddings_results[feat_method] = embeddings
        # Optionally save embeddings for reference
        np.save(os.path.join(output_dir, f"{feat_method}_embeddings.npy"), embeddings)

        for cluster_method in clustering_methods:
            print(f"Clustering using {cluster_method} on {feat_method} embeddings")
            clustering = SpeakerClustering()
            # Perform clustering (automatically determines number of speakers)
            # For AHC, use a threshold of 0.3
            # For DBSCAN, the parameters are automatically determined
            clustering.perform_clustering(embeddings, threshold=0.3)
            vis_path = os.path.join(output_dir, f"clustering_{feat_method}_{cluster_method}.png")
            clustering.visualize_clusters(vis_path)

    print("\nProcessing complete. Check the output folder for all results and plots.")

if __name__ == "__main__":
    main()
