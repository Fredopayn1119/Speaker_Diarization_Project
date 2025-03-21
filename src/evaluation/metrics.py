# This file defines evaluation metrics, such as Diarization Error Rate (DER),
# for assessing the performance of the speaker diarization system.

"""Evaluation metrics for speaker diarization."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import os
import re


@dataclass
class DiarizationErrorRate:
    """Class to store DER components."""
    miss: float = 0.0  # Missed speech
    false_alarm: float = 0.0  # False alarm speech
    confusion: float = 0.0  # Speaker confusion
    total: float = 0.0  # Total DER
    
    def __str__(self) -> str:
        """String representation of DER."""
        return (f"DER: {self.total:.2f}% "
                f"(Miss: {self.miss:.2f}%, "
                f"FA: {self.false_alarm:.2f}%, "
                f"Confusion: {self.confusion:.2f}%)")


def load_rttm_file(rttm_path: str) -> Dict[str, List[Tuple[float, float, int]]]:
    """
    Load RTTM file into a structured format.
    
    Args:
        rttm_path: Path to RTTM file
        
    Returns:
        Dictionary mapping file_id to list of (start, end, speaker_id) tuples
    """
    segments = {}
    
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
                
            file_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker_id = int(parts[7])
            
            if file_id not in segments:
                segments[file_id] = []
                
            segments[file_id].append((start, end, speaker_id))
    
    # Sort segments by start time for each file
    for file_id in segments:
        segments[file_id].sort(key=lambda x: x[0])
    
    return segments


def calculate_der(reference_rttm: str, hypothesis_rttm: str, 
                 collar: float = 0.25, ignore_overlaps: bool = True) -> Dict[str, DiarizationErrorRate]:
    """
    Calculate Diarization Error Rate (DER) between reference and hypothesis RTTMs.
    
    Args:
        reference_rttm: Path to reference RTTM file
        hypothesis_rttm: Path to hypothesis RTTM file
        collar: Collar size in seconds to ignore errors around boundaries
        ignore_overlaps: Whether to ignore overlap regions
        
    Returns:
        Dictionary mapping file_id to DiarizationErrorRate objects
    """
    try:
        # Check if pyannote.metrics is available for accurate DER calculation
        from pyannote.metrics.diarization import DiarizationErrorRate as PyannoteMetricDER
        
        print("Using pyannote.metrics for DER calculation")
        return calculate_der_pyannote(reference_rttm, hypothesis_rttm, collar, ignore_overlaps)
    except ImportError:
        print("pyannote.metrics not found, using simplified DER calculation")
        # Fall back to a simplified implementation
        return calculate_der_simplified(reference_rttm, hypothesis_rttm, collar, ignore_overlaps)


def calculate_der_pyannote(reference_rttm: str, hypothesis_rttm: str, 
                          collar: float = 0.25, ignore_overlaps: bool = True) -> Dict[str, DiarizationErrorRate]:
    """
    Calculate DER using pyannote.metrics.
    
    Args:
        reference_rttm: Path to reference RTTM file
        hypothesis_rttm: Path to hypothesis RTTM file
        collar: Collar size in seconds
        ignore_overlaps: Whether to ignore overlap regions
        
    Returns:
        Dictionary mapping file_id to DiarizationErrorRate objects
    """
    from pyannote.metrics.diarization import DiarizationErrorRate as PyannoteMetricDER
    from pyannote.database.util import load_rttm
    
    # Load reference and hypothesis RTTMs
    reference = load_rttm(reference_rttm)
    hypothesis = load_rttm(hypothesis_rttm)
    
    # Initialize metric
    metric = PyannoteMetricDER(collar=collar, skip_overlap=ignore_overlaps)
    
    results = {}
    file_ids = set(reference.keys()).union(set(hypothesis.keys()))
    
    for file_id in file_ids:
        ref = reference.get(file_id, None)
        hyp = hypothesis.get(file_id, None)
        
        if ref is None or hyp is None:
            print(f"Warning: File {file_id} is missing in {'reference' if ref is None else 'hypothesis'}")
            continue
        
        # Calculate detailed DER
        confusion, missed, false_alarm = metric.compute_components(ref, hyp)
        total = confusion + missed + false_alarm
        
        # Store results
        results[file_id] = DiarizationErrorRate(
            miss=missed * 100,
            false_alarm=false_alarm * 100,
            confusion=confusion * 100,
            total=total * 100
        )
    
    # Calculate average DER across all files
    if results:
        avg_miss = sum(r.miss for r in results.values()) / len(results)
        avg_fa = sum(r.false_alarm for r in results.values()) / len(results)
        avg_conf = sum(r.confusion for r in results.values()) / len(results)
        avg_total = sum(r.total for r in results.values()) / len(results)
        
        results["AVERAGE"] = DiarizationErrorRate(
            miss=avg_miss, 
            false_alarm=avg_fa, 
            confusion=avg_conf, 
            total=avg_total
        )
    
    return results


def calculate_der_simplified(reference_rttm: str, hypothesis_rttm: str, 
                            collar: float = 0.25, ignore_overlaps: bool = True) -> Dict[str, DiarizationErrorRate]:
    """
    Calculate DER with a simplified approach (less accurate but no dependencies).
    
    Args:
        reference_rttm: Path to reference RTTM file
        hypothesis_rttm: Path to hypothesis RTTM file
        collar: Collar size in seconds
        ignore_overlaps: Whether to ignore overlap regions
        
    Returns:
        Dictionary mapping file_id to DiarizationErrorRate objects
    """
    # Load reference and hypothesis RTTMs
    ref_segments = load_rttm_file(reference_rttm)
    hyp_segments = load_rttm_file(hypothesis_rttm)
    
    results = {}
    file_ids = set(ref_segments.keys()).union(set(hyp_segments.keys()))
    
    for file_id in file_ids:
        ref = ref_segments.get(file_id, [])
        hyp = hyp_segments.get(file_id, [])
        
        if not ref or not hyp:
            print(f"Warning: File {file_id} is missing or empty in {'reference' if not ref else 'hypothesis'}")
            continue
        
        # Convert segments to a flat timeline with speaker labels for both reference and hypothesis
        # This is a simplified approach that discretizes time into small frames
        frame_rate = 100  # frames per second (10ms per frame)
        duration = max(max(seg[1] for seg in ref), max(seg[1] for seg in hyp))
        num_frames = int(duration * frame_rate) + 1
        
        ref_timeline = [-1] * num_frames  # -1 means no speech
        hyp_timeline = [-1] * num_frames
        
        # Fill reference timeline
        for start, end, speaker_id in ref:
            start_frame = max(0, int((start + collar) * frame_rate))
            end_frame = min(num_frames, int((end - collar) * frame_rate))
            for i in range(start_frame, end_frame):
                if ignore_overlaps and ref_timeline[i] != -1:
                    ref_timeline[i] = -2  # -2 marks overlap (to be ignored)
                else:
                    ref_timeline[i] = speaker_id
        
        # Fill hypothesis timeline
        for start, end, speaker_id in hyp:
            start_frame = max(0, int((start + collar) * frame_rate))
            end_frame = min(num_frames, int((end - collar) * frame_rate))
            for i in range(start_frame, end_frame):
                if ignore_overlaps and hyp_timeline[i] != -1:
                    hyp_timeline[i] = -2  # overlap in hypothesis
                else:
                    hyp_timeline[i] = speaker_id
        
        # Count frames for each error type
        miss = 0
        false_alarm = 0
        confusion = 0
        total_speech = 0
        
        for i in range(num_frames):
            # Skip frames marked as overlap if ignore_overlaps is True
            if ignore_overlaps and (ref_timeline[i] == -2 or hyp_timeline[i] == -2):
                continue
                
            if ref_timeline[i] != -1:
                total_speech += 1
                
            if ref_timeline[i] != -1 and hyp_timeline[i] == -1:
                miss += 1
            elif ref_timeline[i] == -1 and hyp_timeline[i] != -1:
                false_alarm += 1
            elif ref_timeline[i] != -1 and hyp_timeline[i] != -1 and ref_timeline[i] != hyp_timeline[i]:
                confusion += 1
        
        # Calculate error rates as percentages of total speech frames
        if total_speech > 0:
            miss_rate = (miss / total_speech) * 100
            fa_rate = (false_alarm / total_speech) * 100
            conf_rate = (confusion / total_speech) * 100
            total_der = miss_rate + fa_rate + conf_rate
        else:
            miss_rate = fa_rate = conf_rate = total_der = 0
        
        results[file_id] = DiarizationErrorRate(
            miss=miss_rate,
            false_alarm=fa_rate,
            confusion=conf_rate,
            total=total_der
        )
    
    # Calculate average DER across all files
    if results:
        avg_miss = sum(r.miss for r in results.values()) / len(results)
        avg_fa = sum(r.false_alarm for r in results.values()) / len(results)
        avg_conf = sum(r.confusion for r in results.values()) / len(results)
        avg_total = sum(r.total for r in results.values()) / len(results)
        
        results["AVERAGE"] = DiarizationErrorRate(
            miss=avg_miss, 
            false_alarm=avg_fa, 
            confusion=avg_conf, 
            total=avg_total
        )
    
    return results


def print_der_results(results: Dict[str, DiarizationErrorRate]):
    """
    Print DER results in a formatted table.
    
    Args:
        results: Dictionary mapping file_id to DiarizationErrorRate objects
    """
    print("\n" + "="*80)
    print(f"{'File ID':<30} {'Miss %':<10} {'FA %':<10} {'Confusion %':<15} {'DER %':<10}")
    print("-"*80)
    
    for file_id, der in sorted(results.items()):
        if file_id == "AVERAGE":
            print("-"*80)
        print(f"{file_id:<30} {der.miss:<10.2f} {der.false_alarm:<10.2f} {der.confusion:<15.2f} {der.total:<10.2f}")
    
    print("="*80)
