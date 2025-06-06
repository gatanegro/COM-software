"""
COM Framework Analysis Module Fixes

This module implements fixes for the analysis modules of the Continuous Oscillatory
Model (COM) framework, addressing issues identified in testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from scipy import stats, signal, optimize, fft
from sklearn import cluster, decomposition, manifold
from typing import List, Tuple, Dict, Optional, Union, Callable
import math
import itertools

# Import the core modules
from com_framework_core_fixed import LZModule, OctaveModule

class MathematicalAnalysisModule:
    """
    Advanced mathematical analysis tools for the COM framework.
    
    This module provides functions for analyzing mathematical properties
    of LZ-based systems, octave patterns, and energy distributions.
    """
    
    def __init__(self, lz_module: LZModule = None, octave_module: OctaveModule = None):
        """
        Initialize the Mathematical Analysis module.
        
        Args:
            lz_module: Reference to LZ module
            octave_module: Reference to Octave module
        """
        self.lz_module = lz_module if lz_module else LZModule()
        self.octave_module = octave_module if octave_module else OctaveModule(self.lz_module)
        
    def analyze_fixed_points(self, start: float = 0.0, end: float = 10.0, 
                           step: float = 0.01) -> Dict:
        """
        Analyze all fixed points of the recursive function in a given range.
        
        Args:
            start: Start of search range
            end: End of search range
            step: Step size for search
            
        Returns:
            Dictionary with analysis results
        """
        fixed_points = self.lz_module.find_fixed_points(start, end, step)
        
        # Analyze each fixed point
        results = {
            'fixed_points': fixed_points,
            'count': len(fixed_points),
            'stability': [],
            'basin_sizes': [],
            'convergence_rates': []
        }
        
        for fp in fixed_points:
            # Check stability
            stability = self.lz_module.stability_at_point(fp)
            results['stability'].append(stability)
            
            # Estimate basin of attraction size
            basin_size = self._estimate_basin_size(fp, start, end, step)
            results['basin_sizes'].append(basin_size)
            
            # Calculate convergence rate
            conv_rate = self._calculate_convergence_rate(fp)
            results['convergence_rates'].append(conv_rate)
        
        return results
    
    def _estimate_basin_size(self, fixed_point: float, start: float, end: float, 
                           step: float) -> float:
        """
        Estimate the size of the basin of attraction for a fixed point.
        
        Args:
            fixed_point: The fixed point to analyze
            start: Start of search range
            end: End of search range
            step: Step size for search
            
        Returns:
            Estimated size of basin of attraction
        """
        count = 0
        total = 0
        
        # Use fewer test points for efficiency
        test_points = np.linspace(start, end, int((end - start) / step / 10))
        
        for x in test_points:
            total += 1
            
            # Check if this point converges to the fixed point
            current = x
            for _ in range(50):  # Limit iterations
                current = self.lz_module.recursive_wave_function(current)
                
                # Check if converged to this fixed point
                if abs(current - fixed_point) < 1e-3:
                    count += 1
                    break
        
        # Return proportion of points that converge to this fixed point
        return count / total if total > 0 else 0
    
    def _calculate_convergence_rate(self, fixed_point: float, 
                                  delta: float = 0.01) -> float:
        """
        Calculate the convergence rate near a fixed point.
        
        Args:
            fixed_point: The fixed point to analyze
            delta: Small perturbation for calculation
            
        Returns:
            Convergence rate (iterations needed to halve the distance)
        """
        # Start slightly away from fixed point
        x = fixed_point + delta
        
        # Track distance from fixed point
        distances = []
        
        # Iterate until very close or max iterations
        for _ in range(20):  # Reduced iterations for efficiency
            x = self.lz_module.recursive_wave_function(x)
            dist = abs(x - fixed_point)
            distances.append(dist)
            
            if dist < 1e-6:
                break
        
        # Calculate average rate of convergence
        if len(distances) < 2:
            return 0
        
        rates = []
        for i in range(1, len(distances)):
            if distances[i-1] > 0:  # Avoid division by zero
                rate = distances[i] / distances[i-1]
                rates.append(rate)
        
        # Return average rate
        return np.mean(rates) if rates else 0
    
    def analyze_lz_powers(self, max_power: int = 10) -> Dict:
        """
        Analyze properties of powers of the LZ constant.
        
        Args:
            max_power: Maximum power to analyze
            
        Returns:
            Dictionary with analysis results
        """
        powers = [self.lz_module.LZ ** i for i in range(-max_power, max_power + 1)]
        
        # Analyze octave patterns in powers
        octaves = [self.octave_module.lz_based_octave(p) for p in powers]
        
        # Find patterns in the sequence
        pattern_length = self._find_pattern_length(octaves)
        
        # Calculate ratios between consecutive powers
        ratios = [powers[i+1] / powers[i] for i in range(len(powers) - 1)]
        
        return {
            'powers': powers,
            'octaves': octaves,
            'pattern_length': pattern_length,
            'ratios': ratios
        }
    
    def _find_pattern_length(self, sequence: List[float], 
                           threshold: float = 1e-5) -> int:
        """
        Find the length of a repeating pattern in a sequence.
        
        Args:
            sequence: Sequence to analyze
            threshold: Threshold for considering values equal
            
        Returns:
            Length of the repeating pattern, or 0 if none found
        """
        n = len(sequence)
        
        # Try different pattern lengths
        for length in range(1, n // 2 + 1):
            is_pattern = True
            
            # Check if the pattern repeats
            for i in range(length, min(length * 3, n)):
                if abs(sequence[i] - sequence[i % length]) > threshold:
                    is_pattern = False
                    break
            
            if is_pattern:
                return length
        
        return 0
    
    def analyze_octave_distribution(self, start: int = 1, end: int = 1000) -> Dict:
        """
        Analyze the distribution of octaves in a range of numbers.
        
        Args:
            start: Start of range
            end: End of range
            
        Returns:
            Dictionary with analysis results
        """
        sequence = list(range(start, end + 1))
        octaves = [self.octave_module.octave_reduction(n) for n in sequence]
        
        # Count occurrences of each octave
        counts = {i: octaves.count(i) for i in range(1, 10)}
        
        # Calculate expected uniform distribution
        expected = len(sequence) / 9
        
        # Calculate chi-square statistic
        chi2_stat = sum((counts[i] - expected) ** 2 / expected for i in range(1, 10))
        
        # Calculate p-value (8 degrees of freedom)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 8)
        
        # Check for patterns in the sequence
        autocorr = self._calculate_autocorrelation(octaves)
        
        return {
            'counts': counts,
            'chi2_stat': chi2_stat,
            'p_value': p_value,
            'is_uniform': p_value > 0.05,  # Null hypothesis: distribution is uniform
            'autocorrelation': autocorr
        }
    
    def _calculate_autocorrelation(self, sequence: List[int], 
                                 max_lag: int = 20) -> List[float]:
        """
        Calculate autocorrelation of a sequence for different lags.
        
        Args:
            sequence: Sequence to analyze
            max_lag: Maximum lag to consider
            
        Returns:
            List of autocorrelation values for each lag
        """
        # Convert to numpy array
        arr = np.array(sequence)
        
        # Calculate mean and variance
        mean = np.mean(arr)
        var = np.var(arr)
        
        if var == 0:
            return [0] * max_lag
        
        # Calculate autocorrelation for each lag
        autocorr = []
        n = len(arr)
        
        for lag in range(1, min(max_lag + 1, n)):
            # Calculate correlation between original and lagged sequence
            corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
            autocorr.append(corr)
        
        return autocorr
    
    def analyze_collatz_octave_properties(self, max_n: int = 100, 
                                        key: int = 7) -> Dict:
        """
        Analyze properties of Collatz-Octave transformations.
        
        Args:
            max_n: Maximum starting number to analyze
            key: Key for the transformation
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'cycle_lengths': [],
            'sequence_lengths': [],
            'unique_octaves': [],
            'entropy': []
        }
        
        for n in range(1, max_n + 1):
            # Get transformation
            sequence = self.octave_module.collatz_octave_transform(n, key, 100)
            
            # Record sequence length
            results['sequence_lengths'].append(len(sequence))
            
            # Count unique octaves
            unique = len(set(sequence))
            results['unique_octaves'].append(unique)
            
            # Calculate entropy (randomness measure)
            entropy = self._calculate_entropy(sequence)
            results['entropy'].append(entropy)
            
            # Detect cycle length
            cycle_length = self._detect_cycle_length(sequence)
            results['cycle_lengths'].append(cycle_length)
        
        # Calculate statistics
        stats_results = {
            'avg_sequence_length': np.mean(results['sequence_lengths']),
            'avg_unique_octaves': np.mean(results['unique_octaves']),
            'avg_entropy': np.mean(results['entropy']),
            'avg_cycle_length': np.mean(results['cycle_lengths']),
            'max_sequence_length': max(results['sequence_lengths']),
            'min_sequence_length': min(results['sequence_lengths'])
        }
        
        # Combine results
        results.update(stats_results)
        
        return results
    
    def _calculate_entropy(self, sequence: List[int]) -> float:
        """
        Calculate Shannon entropy of a sequence.
        
        Args:
            sequence: Sequence to analyze
            
        Returns:
            Entropy value
        """
        # Count occurrences of each value
        counts = {}
        for val in sequence:
            counts[val] = counts.get(val, 0) + 1
        
        # Calculate probabilities
        n = len(sequence)
        probabilities = [count / n for count in counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in probabilities)
        
        return entropy
    
    def _detect_cycle_length(self, sequence: List[int]) -> int:
        """
        Detect the length of a cycle in a sequence.
        
        Args:
            sequence: Sequence to analyze
            
        Returns:
            Length of the cycle, or 0 if none detected
        """
        n = len(sequence)
        if n < 4:  # Need at least a few elements to detect cycles
            return 0
            
        # Check for cycles of different lengths
        for length in range(1, min(n // 2, 10) + 1):  # Limit to cycles of length 10 or less
            # Check if the last 'length' elements repeat
            is_cycle = True
            
            for i in range(length):
                if n - length - 1 - i < 0:
                    is_cycle = False
                    break
                
                if sequence[n - 1 - i] != sequence[n - length - 1 - i]:
                    is_cycle = False
                    break
            
            if is_cycle:
                return length
        
        return 0
    
    def analyze_lz_hqs_relationship(self, points: int = 1000) -> Dict:
        """
        Analyze the relationship between LZ and HQS threshold.
        
        Args:
            points: Number of points to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Generate values around LZ
        x_values = np.linspace(self.lz_module.LZ * 0.5, self.lz_module.LZ * 1.5, points)
        
        # Calculate function values
        f_values = [self.lz_module.recursive_wave_function(x) for x in x_values]
        
        # Calculate derivatives
        derivatives = [self.lz_module.stability_at_point(x) for x in x_values]
        
        # Find points where derivative crosses 1 (stability boundary)
        stability_crossings = []
        for i in range(1, len(derivatives)):
            if (derivatives[i-1] < 1 and derivatives[i] >= 1) or (derivatives[i-1] >= 1 and derivatives[i] < 1):
                # Linear interpolation to find crossing point
                x_cross = x_values[i-1] + (x_values[i] - x_values[i-1]) * (1 - derivatives[i-1]) / (derivatives[i] - derivatives[i-1])
                stability_crossings.append(x_cross)
        
        # Calculate HQS-related metrics
        hqs_values = [x * 0.235 for x in x_values]  # HQS = 23.5% of x
        
        # Find relationship between HQS and stability
        hqs_stability_corr = np.corrcoef(hqs_values, derivatives)[0, 1]
        
        return {
            'x_values': x_values.tolist(),  # Convert to list for serialization
            'f_values': f_values,
            'derivatives': derivatives,
            'stability_crossings': stability_crossings,
            'hqs_values': hqs_values,
            'hqs_stability_correlation': hqs_stability_corr
        }
    
    def analyze_octave_scaling(self, base: float = 1.0, 
                             octaves: int = 8) -> Dict:
        """
        Analyze scaling relationships across octaves.
        
        Args:
            base: Base value
            octaves: Number of octaves to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Generate scaled values
        scaled_values = [base * (self.lz_module.LZ ** i) for i in range(octaves)]
        
        # Calculate ratios between consecutive octaves
        ratios = [scaled_values[i+1] / scaled_values[i] for i in range(octaves - 1)]
        
        # Calculate octave values
        octave_values = [self.octave_module.octave_reduction(int(val * 100)) for val in scaled_values]
        
        # Check for patterns in octave values
        pattern_length = self._find_pattern_length(octave_values)
        
        return {
            'scaled_values': scaled_values,
            'ratios': ratios,
            'octave_values': octave_values,
            'pattern_length': pattern_length
        }


class PatternRecognitionModule:
    """
    Pattern recognition and analysis tools for the COM framework.
    
    This module provides functions for detecting and analyzing patterns
    in data using COM principles.
    """
    
    def __init__(self, lz_module: LZModule = None, octave_module: OctaveModule = None):
        """
        Initialize the Pattern Recognition module.
        
        Args:
            lz_module: Reference to LZ module
            octave_module: Reference to Octave module
        """
        self.lz_module = lz_module if lz_module else LZModule()
        self.octave_module = octave_module if octave_module else OctaveModule(self.lz_module)
        
    def detect_octave_patterns(self, data: List[int]) -> Dict:
        """
        Detect octave patterns in a sequence of numbers.
        
        Args:
            data: Sequence of numbers to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        # Convert to octaves
        octaves = [self.octave_module.octave_reduction(n) for n in data]
        
        # Find repeating subsequences
        patterns = self._find_repeating_subsequences(octaves)
        
        # Calculate frequency distribution
        freq_dist = {i: octaves.count(i) for i in range(1, 10)}
        
        # Calculate transition matrix
        transitions = self._calculate_transition_matrix(octaves)
        
        # Detect rhythm patterns
        rhythm = self._detect_rhythm_pattern(octaves)
        
        return {
            'octaves': octaves,
            'patterns': patterns,
            'frequency_distribution': freq_dist,
            'transition_matrix': transitions,
            'rhythm': rhythm
        }
    
    def _find_repeating_subsequences(self, sequence: List[int], 
                                   min_length: int = 2, 
                                   min_occurrences: int = 2) -> List[Dict]:
        """
        Find repeating subsequences in a sequence.
        
        Args:
            sequence: Sequence to analyze
            min_length: Minimum length of subsequences to consider
            min_occurrences: Minimum number of occurrences to report
            
        Returns:
            List of dictionaries with pattern information
        """
        n = len(sequence)
        patterns = []
        
        # Check subsequences of different lengths
        for length in range(min_length, min(20, n // 2 + 1)):
            # Dictionary to store subsequences and their positions
            subsequences = {}
            
            # Find all subsequences of the current length
            for i in range(n - length + 1):
                subseq = tuple(sequence[i:i+length])
                if subseq in subsequences:
                    subsequences[subseq].append(i)
                else:
                    subsequences[subseq] = [i]
            
            # Filter subsequences that appear multiple times
            for subseq, positions in subsequences.items():
                if len(positions) >= min_occurrences:
                    patterns.append({
                        'pattern': list(subseq),
                        'length': length,
                        'occurrences': len(positions),
                        'positions': positions
                    })
        
        # Sort by number of occurrences (descending)
        patterns.sort(key=lambda x: x['occurrences'], reverse=True)
        
        return patterns
    
    def _calculate_transition_matrix(self, sequence: List[int]) -> List[List[float]]:
        """
        Calculate transition probabilities between octaves.
        
        Args:
            sequence: Sequence of octaves
            
        Returns:
            9x9 transition matrix (0-indexed, but octave 0 is not used)
        """
        # Initialize 9x9 matrix with zeros
        matrix = [[0.0 for _ in range(9)] for _ in range(9)]
        
        # Count transitions
        for i in range(len(sequence) - 1):
            from_octave = sequence[i] - 1  # Convert to 0-indexed
            to_octave = sequence[i+1] - 1  # Convert to 0-indexed
            matrix[from_octave][to_octave] += 1
        
        # Convert counts to probabilities
        for i in range(9):
            row_sum = sum(matrix[i])
            if row_sum > 0:
                matrix[i] = [count / row_sum for count in matrix[i]]
        
        return matrix
    
    def _detect_rhythm_pattern(self, sequence: List[int]) -> Dict:
        """
        Detect rhythmic patterns in a sequence of octaves.
        
        Args:
            sequence: Sequence of octaves
            
        Returns:
            Dictionary with rhythm analysis
        """
        if len(sequence) < 2:
            return {
                'differences': [],
                'most_common_diff': (0, 0),
                'alternating_pattern': False,
                'increasing': False,
                'decreasing': False
            }
            
        # Calculate differences between consecutive octaves
        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence) - 1)]
        
        # Count occurrences of each difference
        diff_counts = {}
        for diff in diffs:
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
        
        # Calculate most common difference
        most_common_diff = max(diff_counts.items(), key=lambda x: x[1]) if diff_counts else (0, 0)
        
        # Detect alternating patterns (e.g., +2, -2, +2, -2)
        alternating = False
        if len(diffs) >= 4:
            # Check if even-indexed and odd-indexed differences are consistent
            even_diffs = diffs[0::2]
            odd_diffs = diffs[1::2]
            
            even_consistent = all(d == even_diffs[0] for d in even_diffs)
            odd_consistent = all(d == odd_diffs[0] for d in odd_diffs)
            
            alternating = even_consistent and odd_consistent and even_diffs[0] != odd_diffs[0]
        
        # Detect if the sequence follows a scale-like pattern (consistently increasing or decreasing)
        increasing = all(d > 0 for d in diffs)
        decreasing = all(d < 0 for d in diffs)
        
        return {
            'differences': diffs,
            'most_common_diff': most_common_diff,
            'alternating_pattern': alternating,
            'increasing': increasing,
            'decreasing': decreasing
        }
    
    def cluster_by_octave_pattern(self, data_points: List[List[int]]) -> Dict:
        """
        Cluster data points based on their octave patterns.
        
        Args:
            data_points: List of sequences to cluster
            
        Returns:
            Dictionary with clustering results
        """
        if not data_points or len(data_points) < 2:
            return {
                'optimal_clusters': 1,
                'labels': [0] * len(data_points),
                'centers': [[0] * (len(data_points[0]) if data_points else 0)],
                'silhouette_score': 0,
                'inertia': [0]
            }
            
        # Convert each sequence to its octave representation
        octave_sequences = []
        for sequence in data_points:
            octaves = [self.octave_module.octave_reduction(n) for n in sequence]
            octave_sequences.append(octaves)
        
        # Ensure all sequences have the same length
        min_length = min(len(seq) for seq in octave_sequences)
        octave_sequences = [seq[:min_length] for seq in octave_sequences]
        
        # Convert to feature vectors
        features = np.array(octave_sequences)
        
        # Determine optimal number of clusters
        max_clusters = min(10, len(data_points))
        inertia = []
        
        for k in range(1, max_clusters + 1):
            kmeans = cluster.KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features)
            inertia.append(kmeans.inertia_)
        
        # Find elbow point
        optimal_k = self._find_elbow_point(inertia) + 1
        
        # Perform clustering with optimal k
        kmeans = cluster.KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Calculate cluster centers
        centers = kmeans.cluster_centers_
        
        # Calculate silhouette score
        silhouette = 0
        if optimal_k > 1 and optimal_k < len(data_points):
            try:
                silhouette = cluster.silhouette_score(features, labels)
            except:
                silhouette = 0
        
        return {
            'optimal_clusters': optimal_k,
            'labels': labels.tolist(),  # Convert to list for serialization
            'centers': centers.tolist(),  # Convert to list for serialization
            'silhouette_score': silhouette,
            'inertia': inertia
        }
    
    def _find_elbow_point(self, inertia: List[float]) -> int:
        """
        Find the elbow point in the inertia curve for K-means.
        
        Args:
            inertia: List of inertia values
            
        Returns:
            Index of the elbow point
        """
        if len(inertia) <= 2:
            return 0
        
        # Calculate the angle at each point
        angles = []
        for i in range(1, len(inertia) - 1):
            # Create vectors from point i to points i-1 and i+1
            v1 = np.array([1, inertia[i-1] - inertia[i]])
            v2 = np.array([1, inertia[i+1] - inertia[i]])
            
            # Normalize vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm == 0 or v2_norm == 0:
                angles.append(0)
                continue
                
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            # Calculate angle
            dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            angles.append(angle)
        
        # Return index of maximum angle
        return np.argmax(angles) if angles else 0
    
    def detect_lz_based_patterns(self, data: List[float]) -> Dict:
        """
        Detect patterns based on LZ scaling in continuous data.
        
        Args:
            data: Sequence of values to analyze
            
        Returns:
            Dictionary with pattern analysis results
        """
        if not data:
            return {
                'autocorrelation': [],
                'peaks': [],
                'dominant_frequencies': [],
                'lz_related_periods': []
            }
            
        # Calculate autocorrelation
        autocorr = self._calculate_autocorrelation(data)
        
        # Find peaks in autocorrelation
        peaks = []
        try:
            peaks, _ = signal.find_peaks(autocorr)
            peaks = peaks.tolist()  # Convert to list for serialization
        except:
            peaks = []
        
        # Calculate Fourier transform
        fft_result = np.abs(fft.fft(data))
        freqs = fft.fftfreq(len(data))
        
        # Find dominant frequencies
        dominant_indices = np.argsort(fft_result)[-5:]  # Top 5 frequencies
        dominant_freqs = freqs[dominant_indices]
        dominant_magnitudes = fft_result[dominant_indices]
        
        # Check for LZ-related periods
        lz_periods = []
        for i, peak in enumerate(peaks):
            period = peak + 1  # Convert index to period
            
            # Check if period is close to a power of LZ
            for power in range(-5, 6):
                expected = self.lz_module.LZ ** power
                if abs(period - expected) / expected < 0.1:  # Within 10%
                    lz_periods.append({
                        'period': period,
                        'expected': expected,
                        'power': power,
                        'error': abs(period - expected) / expected
                    })
                    break
        
        return {
            'autocorrelation': autocorr,
            'peaks': peaks,
            'dominant_frequencies': [(float(f), float(m)) for f, m in zip(dominant_freqs, dominant_magnitudes)],
            'lz_related_periods': lz_periods
        }
    
    def _calculate_autocorrelation(self, sequence: List[float], 
                                 max_lag: int = None) -> List[float]:
        """
        Calculate autocorrelation of a sequence for different lags.
        
        Args:
            sequence: Sequence to analyze
            max_lag: Maximum lag to consider (default: half of sequence length)
            
        Returns:
            List of autocorrelation values for each lag
        """
        # Convert to numpy array
        arr = np.array(sequence)
        
        # Set default max_lag
        if max_lag is None:
            max_lag = len(arr) // 2
        
        # Calculate autocorrelation
        result = np.correlate(arr, arr, mode='full')
        
        # Extract the relevant part (positive lags)
        result = result[len(arr)-1:len(arr)+max_lag]
        
        # Normalize
        if result[0] != 0:
            result = result / result[0]
        
        return result.tolist()
    
    def find_octave_based_features(self, data: List[int]) -> Dict:
        """
        Extract octave-based features from a sequence.
        
        Args:
            data: Sequence of numbers to analyze
            
        Returns:
            Dictionary with extracted features
        """
        if not data:
            return {
                'mean': 0,
                'median': 0,
                'mode': 0,
                'std_dev': 0,
                'frequency_distribution': {i: 0 for i in range(1, 10)},
                'entropy': 0,
                'runs': {'max_run': 0, 'avg_run': 0, 'run_counts': {}},
                'transitions': {'direction_changes': 0, 'avg_step': 0, 'steps': []}
            }
            
        # Convert to octaves
        octaves = [self.octave_module.octave_reduction(n) for n in data]
        
        # Calculate basic statistics
        mean = float(np.mean(octaves))
        median = float(np.median(octaves))
        
        # Calculate mode safely
        try:
            mode_result = stats.mode(octaves)
            if hasattr(mode_result, 'mode'):
                mode = float(mode_result.mode[0])
            else:
                mode = float(mode_result[0][0])
        except:
            # Fallback if stats.mode fails
            counts = {}
            for o in octaves:
                counts[o] = counts.get(o, 0) + 1
            mode = max(counts.items(), key=lambda x: x[1])[0] if counts else 0
            
        std_dev = float(np.std(octaves))
        
        # Calculate frequency distribution
        freq_dist = {i: octaves.count(i) / len(octaves) for i in range(1, 10)}
        
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in freq_dist.values() if p > 0)
        
        # Calculate run lengths
        runs = self._calculate_run_lengths(octaves)
        
        # Calculate transition features
        transitions = self._calculate_transition_features(octaves)
        
        return {
            'mean': mean,
            'median': median,
            'mode': mode,
            'std_dev': std_dev,
            'frequency_distribution': freq_dist,
            'entropy': entropy,
            'runs': runs,
            'transitions': transitions
        }
    
    def _calculate_run_lengths(self, sequence: List[int]) -> Dict:
        """
        Calculate statistics about runs of the same value.
        
        Args:
            sequence: Sequence to analyze
            
        Returns:
            Dictionary with run length statistics
        """
        if not sequence:
            return {'max_run': 0, 'avg_run': 0, 'run_counts': {}}
        
        # Find runs
        runs = []
        current_run = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        
        # Add the last run
        runs.append(current_run)
        
        # Calculate statistics
        max_run = max(runs)
        avg_run = float(np.mean(runs))
        
        # Count occurrences of each run length
        run_counts = {}
        for run in runs:
            run_counts[run] = run_counts.get(run, 0) + 1
        
        return {
            'max_run': max_run,
            'avg_run': avg_run,
            'run_counts': run_counts
        }
    
    def _calculate_transition_features(self, sequence: List[int]) -> Dict:
        """
        Calculate features based on transitions between values.
        
        Args:
            sequence: Sequence to analyze
            
        Returns:
            Dictionary with transition features
        """
        if len(sequence) < 2:
            return {'direction_changes': 0, 'avg_step': 0, 'steps': []}
        
        # Calculate steps between consecutive values
        steps = [sequence[i+1] - sequence[i] for i in range(len(sequence) - 1)]
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(steps)):
            if (steps[i] > 0 and steps[i-1] <= 0) or (steps[i] < 0 and steps[i-1] >= 0):
                direction_changes += 1
        
        # Calculate average absolute step
        avg_step = float(np.mean([abs(step) for step in steps]))
        
        return {
            'direction_changes': direction_changes,
            'avg_step': avg_step,
            'steps': steps
        }


class StatisticalAnalysisModule:
    """
    Statistical analysis tools for the COM framework.
    
    This module provides functions for statistical analysis of
    COM-related data and patterns.
    """
    
    def __init__(self, lz_module: LZModule = None, octave_module: OctaveModule = None):
        """
        Initialize the Statistical Analysis module.
        
        Args:
            lz_module: Reference to LZ module
            octave_module: Reference to Octave module
        """
        self.lz_module = lz_module if lz_module else LZModule()
        self.octave_module = octave_module if octave_module else OctaveModule(self.lz_module)
        
    def analyze_octave_randomness(self, data: List[int], 
                                significance: float = 0.05) -> Dict:
        """
        Analyze the randomness of octave patterns.
        
        Args:
            data: Sequence of numbers to analyze
            significance: Significance level for statistical tests
            
        Returns:
            Dictionary with randomness analysis results
        """
        if not data:
            return {
                'runs_test': {
                    'statistic': 0,
                    'p_value': 0,
                    'is_random': False
                },
                'chi_square_test': {
                    'statistic': 0,
                    'p_value': 0,
                    'is_uniform': False
                },
                'autocorrelation': {
                    'values': [],
                    'is_significant': False
                },
                'entropy': {
                    'value': 0,
                    'normalized': 0,
                    'is_high': False
                },
                'overall_assessment': {
                    'is_random': False,
                    'confidence': 0
                }
            }
            
        # Convert to octaves
        octaves = [self.octave_module.octave_reduction(n) for n in data]
        
        # Runs test for randomness
        runs_pvalue = 0
        runs = 0
        
        try:
            # Create binary sequence for runs test
            median_val = np.median(octaves)
            binary_seq = [1 if x > median_val else 0 for x in octaves]
            
            # Count runs
            runs_count = 1
            for i in range(1, len(binary_seq)):
                if binary_seq[i] != binary_seq[i-1]:
                    runs_count += 1
                    
            # Count number of each value
            n1 = binary_seq.count(1)
            n0 = binary_seq.count(0)
            
            # Calculate expected number of runs
            n = n0 + n1
            expected_runs = (2 * n0 * n1) / n + 1
            
            # Calculate variance
            var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1))
            
            # Calculate z-statistic
            if var_runs > 0:
                z = (runs_count - expected_runs) / np.sqrt(var_runs)
                runs = z
                
                # Calculate p-value
                runs_pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                runs = 0
                runs_pvalue = 1.0
        except:
            # Fallback if runs test calculation fails
            runs = 0
            runs_pvalue = 0
        
        # Chi-square test for uniform distribution
        observed = [octaves.count(i) for i in range(1, 10)]
        expected = [len(octaves) / 9] * 9
        
        chi2 = 0
        chi2_pvalue = 0
        
        try:
            chi2, chi2_pvalue = stats.chisquare(observed, expected)
        except:
            # Fallback if chi-square test fails
            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
            chi2_pvalue = 0
        
        # Autocorrelation test
        autocorr = self._calculate_autocorrelation(octaves, 10)
        threshold = 2 / np.sqrt(len(octaves))
        autocorr_significant = any(abs(ac) > threshold for ac in autocorr)
        
        # Entropy calculation
        counts = {i: octaves.count(i) for i in range(1, 10)}
        probabilities = [count / len(octaves) for count in counts.values() if count > 0]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        max_entropy = math.log2(9)  # Maximum possible entropy for 9 octaves
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Overall randomness assessment
        is_random = (
            runs_pvalue > significance and
            chi2_pvalue > significance and
            not autocorr_significant and
            normalized_entropy > 0.9
        )
        
        # Calculate confidence
        confidence = self._calculate_randomness_confidence(
            runs_pvalue, chi2_pvalue, autocorr_significant, normalized_entropy
        )
        
        return {
            'runs_test': {
                'statistic': float(runs),
                'p_value': float(runs_pvalue),
                'is_random': runs_pvalue > significance
            },
            'chi_square_test': {
                'statistic': float(chi2),
                'p_value': float(chi2_pvalue),
                'is_uniform': chi2_pvalue > significance
            },
            'autocorrelation': {
                'values': autocorr,
                'is_significant': autocorr_significant
            },
            'entropy': {
                'value': float(entropy),
                'normalized': float(normalized_entropy),
                'is_high': normalized_entropy > 0.9
            },
            'overall_assessment': {
                'is_random': is_random,
                'confidence': float(confidence)
            }
        }
    
    def _calculate_autocorrelation(self, sequence: List[int], 
                                 max_lag: int = 10) -> List[float]:
        """
        Calculate autocorrelation of a sequence for different lags.
        
        Args:
            sequence: Sequence to analyze
            max_lag: Maximum lag to consider
            
        Returns:
            List of autocorrelation values for each lag
        """
        # Convert to numpy array
        arr = np.array(sequence)
        
        # Calculate mean and variance
        mean = np.mean(arr)
        var = np.var(arr)
        
        if var == 0:
            return [0] * max_lag
        
        # Calculate autocorrelation for each lag
        autocorr = []
        n = len(arr)
        
        for lag in range(1, min(max_lag + 1, n)):
            # Calculate correlation between original and lagged sequence
            try:
                corr = np.corrcoef(arr[:-lag], arr[lag:])[0, 1]
                autocorr.append(float(corr))
            except:
                autocorr.append(0.0)
        
        return autocorr
    
    def _calculate_randomness_confidence(self, runs_pvalue: float, 
                                       chi2_pvalue: float, 
                                       autocorr_significant: bool, 
                                       normalized_entropy: float) -> float:
        """
        Calculate confidence in randomness assessment.
        
        Args:
            runs_pvalue: P-value from runs test
            chi2_pvalue: P-value from chi-square test
            autocorr_significant: Whether autocorrelation is significant
            normalized_entropy: Normalized entropy value
            
        Returns:
            Confidence score (0-1)
        """
        # Convert p-values to confidence scores (higher p-value = higher confidence in randomness)
        runs_conf = min(runs_pvalue * 2, 1.0)
        chi2_conf = min(chi2_pvalue * 2, 1.0)
        
        # Convert autocorrelation to confidence score
        autocorr_conf = 0.0 if autocorr_significant else 1.0
        
        # Entropy is already normalized to 0-1
        entropy_conf = normalized_entropy
        
        # Weighted average of confidence scores
        weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each test
        confidence = (
            weights[0] * runs_conf +
            weights[1] * chi2_conf +
            weights[2] * autocorr_conf +
            weights[3] * entropy_conf
        )
        
        return confidence
    
    def analyze_crypto_security(self, key: int, 
                              sample_size: int = 100, 
                              iterations: int = 10) -> Dict:
        """
        Analyze the security properties of the Collatz-Octave cryptography.
        
        Args:
            key: Encryption key to analyze
            sample_size: Number of test cases
            iterations: Number of encryption iterations
            
        Returns:
            Dictionary with security analysis results
        """
        # Generate test cases
        test_cases = list(range(1, sample_size + 1))
        
        # Encrypt each test case
        encrypted = [self.octave_module.collatz_octave_transform(n, key, iterations) for n in test_cases]
        
        # Analyze avalanche effect (how small changes in input affect output)
        avalanche = self._analyze_avalanche_effect(key, iterations)
        
        # Analyze key sensitivity
        key_sensitivity = self._analyze_key_sensitivity(test_cases[0], key, iterations)
        
        # Analyze statistical properties of encrypted data
        flat_encrypted = [val for seq in encrypted for val in seq]
        
        # Count occurrences of each octave
        octave_counts = {i: flat_encrypted.count(i) for i in range(1, 10)}
        
        # Calculate chi-square statistic for uniform distribution
        expected = len(flat_encrypted) / 9
        chi2_stat = sum((octave_counts[i] - expected) ** 2 / expected for i in range(1, 10))
        
        # Calculate p-value (8 degrees of freedom)
        p_value = 1 - stats.chi2.cdf(chi2_stat, 8)
        
        # Calculate entropy
        probabilities = [count / len(flat_encrypted) for count in octave_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        max_entropy = math.log2(9)  # Maximum possible entropy for 9 octaves
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Overall security assessment
        security_score = self._calculate_security_score(
            avalanche['average_difference'],
            key_sensitivity['average_difference'],
            p_value,
            normalized_entropy
        )
        
        return {
            'avalanche_effect': avalanche,
            'key_sensitivity': key_sensitivity,
            'distribution_analysis': {
                'octave_counts': octave_counts,
                'chi2_stat': float(chi2_stat),
                'p_value': float(p_value),
                'is_uniform': p_value > 0.05
            },
            'entropy': {
                'value': float(entropy),
                'normalized': float(normalized_entropy),
                'is_high': normalized_entropy > 0.9
            },
            'security_assessment': {
                'score': float(security_score),
                'rating': self._get_security_rating(security_score)
            }
        }
    
    def _analyze_avalanche_effect(self, key: int, iterations: int) -> Dict:
        """
        Analyze the avalanche effect (how small changes in input affect output).
        
        Args:
            key: Encryption key
            iterations: Number of encryption iterations
            
        Returns:
            Dictionary with avalanche analysis results
        """
        # Test pairs of inputs that differ by 1
        test_pairs = [(i, i+1) for i in range(1, 20)]  # Reduced for efficiency
        
        # Calculate differences in output for each pair
        differences = []
        
        for n1, n2 in test_pairs:
            # Encrypt both inputs
            enc1 = self.octave_module.collatz_octave_transform(n1, key, iterations)
            enc2 = self.octave_module.collatz_octave_transform(n2, key, iterations)
            
            # Calculate difference (using minimum length)
            min_len = min(len(enc1), len(enc2))
            diff = sum(1 for i in range(min_len) if enc1[i] != enc2[i])
            
            # Normalize by length
            norm_diff = diff / min_len if min_len > 0 else 0
            differences.append(norm_diff)
        
        # Calculate statistics
        avg_diff = float(np.mean(differences))
        min_diff = float(np.min(differences))
        max_diff = float(np.max(differences))
        
        return {
            'differences': differences,
            'average_difference': avg_diff,
            'min_difference': min_diff,
            'max_difference': max_diff,
            'is_good': avg_diff > 0.5  # Good avalanche effect should change at least half the output
        }
    
    def _analyze_key_sensitivity(self, input_value: int, key: int, 
                               iterations: int) -> Dict:
        """
        Analyze sensitivity to small changes in the key.
        
        Args:
            input_value: Input value to encrypt
            key: Base encryption key
            iterations: Number of encryption iterations
            
        Returns:
            Dictionary with key sensitivity analysis results
        """
        # Test keys that differ from the base key by small amounts
        test_keys = [key + delta for delta in range(-3, 4) if delta != 0]  # Reduced for efficiency
        
        # Encrypt with base key
        base_enc = self.octave_module.collatz_octave_transform(input_value, key, iterations)
        
        # Calculate differences for each test key
        differences = []
        
        for test_key in test_keys:
            # Encrypt with test key
            test_enc = self.octave_module.collatz_octave_transform(input_value, test_key, iterations)
            
            # Calculate difference (using minimum length)
            min_len = min(len(base_enc), len(test_enc))
            diff = sum(1 for i in range(min_len) if base_enc[i] != test_enc[i])
            
            # Normalize by length
            norm_diff = diff / min_len if min_len > 0 else 0
            differences.append(norm_diff)
        
        # Calculate statistics
        avg_diff = float(np.mean(differences))
        min_diff = float(np.min(differences))
        max_diff = float(np.max(differences))
        
        return {
            'differences': differences,
            'average_difference': avg_diff,
            'min_difference': min_diff,
            'max_difference': max_diff,
            'is_good': avg_diff > 0.5  # Good key sensitivity should change at least half the output
        }
    
    def _calculate_security_score(self, avalanche_score: float, 
                                key_sensitivity_score: float, 
                                uniformity_pvalue: float, 
                                entropy_score: float) -> float:
        """
        Calculate overall security score.
        
        Args:
            avalanche_score: Score for avalanche effect
            key_sensitivity_score: Score for key sensitivity
            uniformity_pvalue: P-value for uniformity test
            entropy_score: Normalized entropy score
            
        Returns:
            Security score (0-100)
        """
        # Convert p-value to score (higher p-value = better uniformity)
        uniformity_score = min(uniformity_pvalue * 2, 1.0)
        
        # Weighted average of scores
        weights = [0.3, 0.3, 0.2, 0.2]  # Weights for each component
        score = (
            weights[0] * avalanche_score +
            weights[1] * key_sensitivity_score +
            weights[2] * uniformity_score +
            weights[3] * entropy_score
        )
        
        # Scale to 0-100
        return score * 100
    
    def _get_security_rating(self, score: float) -> str:
        """
        Convert security score to rating.
        
        Args:
            score: Security score (0-100)
            
        Returns:
            Security rating (text)
        """
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 50:
            return "Moderate"
        else:
            return "Poor"
    
    def analyze_lz_distribution(self, samples: int = 1000, 
                              iterations: int = 100) -> Dict:
        """
        Analyze the distribution of values after repeated application of the LZ function.
        
        Args:
            samples: Number of random starting points
            iterations: Number of iterations to apply
            
        Returns:
            Dictionary with distribution analysis results
        """
        # Generate random starting points
        start_points = np.random.uniform(0.1, 5.0, samples)
        
        # Apply iterations
        end_points = []
        
        for start in start_points:
            current = start
            for _ in range(iterations):
                current = self.lz_module.recursive_wave_function(current)
            end_points.append(current)
        
        # Calculate statistics
        mean = float(np.mean(end_points))
        median = float(np.median(end_points))
        std_dev = float(np.std(end_points))
        
        # Calculate distance from LZ
        distances = [abs(point - self.lz_module.LZ) for point in end_points]
        avg_distance = float(np.mean(distances))
        max_distance = float(np.max(distances))
        
        # Count points within different thresholds of LZ
        thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        counts = [sum(1 for d in distances if d < threshold) for threshold in thresholds]
        percentages = [count / samples * 100 for count in counts]
        
        # Test if distribution is normal
        normality_pvalue = 0
        try:
            _, normality_pvalue = stats.normaltest(end_points)
        except:
            normality_pvalue = 0
        
        return {
            'statistics': {
                'mean': mean,
                'median': median,
                'std_dev': std_dev,
                'proximity_to_lz': {
                    'average_distance': avg_distance,
                    'max_distance': max_distance,
                    'within_thresholds': dict(zip([str(t) for t in thresholds], percentages))
                }
            },
            'normality_test': {
                'p_value': float(normality_pvalue),
                'is_normal': normality_pvalue > 0.05
            },
            'convergence_assessment': {
                'converged_percentage': percentages[-1],  # Percentage within smallest threshold
                'is_convergent': percentages[-1] > 95  # Consider convergent if >95% within smallest threshold
            }
        }
