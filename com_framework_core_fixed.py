"""
COM Framework Core Module Fixes

This module implements fixes for the core functionality of the Continuous Oscillatory
Model (COM) framework, addressing issues identified in testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Tuple, Dict, Optional, Union, Callable
import math

class LZModule:
    """
    Core module for LZ-related calculations and functions.
    
    This module provides functions for working with the LZ constant,
    recursive wave functions, and related mathematical operations.
    """
    
    def __init__(self):
        """Initialize the LZ module with constants."""
        self.LZ = 1.23498  # The LZ constant
        self.HQS = 0.235 * self.LZ  # Harmonic Quantum Scalar (23.5% of LZ)
        
    def recursive_wave_function(self, x: float) -> float:
        """
        Calculate the recursive wave function value at a point.
        
        Args:
            x: Input value
            
        Returns:
            Result of sin(x) + e^(-x)
        """
        return np.sin(x) + np.exp(-x)
    
    def derive_lz(self, initial_value: float = 1.0, 
                max_iterations: int = 100, 
                precision: float = 1e-6) -> Tuple[float, List[float], int]:
        """
        Derive the LZ constant through iteration.
        
        Args:
            initial_value: Starting value for iteration
            max_iterations: Maximum number of iterations
            precision: Convergence threshold
            
        Returns:
            Tuple of (derived LZ value, sequence of iterations, number of iterations)
        """
        sequence = [initial_value]
        current = initial_value
        
        for i in range(max_iterations):
            next_val = self.recursive_wave_function(current)
            sequence.append(next_val)
            
            # Check for convergence
            if abs(next_val - current) < precision:
                return next_val, sequence, i + 1
                
            current = next_val
            
        # If max iterations reached without convergence
        return current, sequence, max_iterations
    
    def verify_lz(self, precision: float = 1e-6) -> bool:
        """
        Verify that LZ is a fixed point of the recursive function.
        
        Args:
            precision: Threshold for verification
            
        Returns:
            True if LZ is a fixed point, False otherwise
        """
        result = self.recursive_wave_function(self.LZ)
        return abs(result - self.LZ) < precision
    
    def stability_at_point(self, x: float) -> float:
        """
        Calculate the stability (magnitude of derivative) at a point.
        
        Args:
            x: Point to evaluate stability
            
        Returns:
            Absolute value of derivative at x
        """
        # Derivative of sin(x) + e^(-x) is cos(x) - e^(-x)
        derivative = np.cos(x) - np.exp(-x)
        return abs(derivative)
    
    def is_stable_fixed_point(self, x: float, 
                            precision: float = 1e-6) -> bool:
        """
        Check if a point is a stable fixed point.
        
        Args:
            x: Point to check
            precision: Threshold for fixed point verification
            
        Returns:
            True if x is a stable fixed point, False otherwise
        """
        # Check if it's a fixed point
        is_fixed = abs(self.recursive_wave_function(x) - x) < precision
        
        # Check stability (derivative magnitude < 1)
        is_stable = self.stability_at_point(x) < 1
        
        return is_fixed and is_stable
    
    def lz_scaling(self, value: float, octave: int) -> float:
        """
        Scale a value by a power of LZ.
        
        Args:
            value: Base value to scale
            octave: Power of LZ to use (can be negative)
            
        Returns:
            Scaled value
        """
        return value * (self.LZ ** octave)
    
    def find_fixed_points(self, start: float, end: float, 
                        step: float) -> List[float]:
        """
        Find fixed points of the recursive function in a range.
        
        Args:
            start: Start of search range
            end: End of search range
            step: Step size for search
            
        Returns:
            List of fixed points found
        """
        fixed_points = []
        
        # Search through the range
        x = start
        while x <= end:
            # Calculate function value
            fx = self.recursive_wave_function(x)
            
            # Check if it's a fixed point
            if abs(fx - x) < step:
                # Refine the fixed point using binary search
                refined = self._refine_fixed_point(x - step, x + step)
                fixed_points.append(refined)
                
                # Skip ahead to avoid finding the same fixed point again
                x += step * 10
            else:
                x += step
                
        return fixed_points
    
    def _refine_fixed_point(self, left: float, right: float, 
                          iterations: int = 10) -> float:
        """
        Refine a fixed point using binary search.
        
        Args:
            left: Left bound
            right: Right bound
            iterations: Number of refinement iterations
            
        Returns:
            Refined fixed point
        """
        for _ in range(iterations):
            mid = (left + right) / 2
            fmid = self.recursive_wave_function(mid)
            
            if fmid > mid:
                left = mid
            else:
                right = mid
                
        return (left + right) / 2
    
    def plot_recursive_function(self, x_min: float = 0, 
                              x_max: float = 3, 
                              points: int = 1000) -> plt.Figure:
        """
        Plot the recursive function and identity line.
        
        Args:
            x_min: Minimum x value
            x_max: Maximum x value
            points: Number of points to plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate x values
        x = np.linspace(x_min, x_max, points)
        
        # Calculate function values
        y = np.array([self.recursive_wave_function(xi) for xi in x])
        
        # Plot function
        ax.plot(x, y, 'b-', label='sin(x) + e^(-x)')
        
        # Plot identity line
        ax.plot(x, x, 'r--', label='y = x')
        
        # Mark LZ
        ax.plot([self.LZ], [self.LZ], 'go', markersize=8, label=f'LZ = {self.LZ:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Recursive Wave Function')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def plot_convergence(self, initial_values: List[float] = None, 
                       max_iterations: int = 20) -> plt.Figure:
        """
        Plot convergence to LZ from different starting points.
        
        Args:
            initial_values: List of starting values
            max_iterations: Maximum number of iterations
            
        Returns:
            Matplotlib figure
        """
        if initial_values is None:
            initial_values = [0.5, 1.0, 1.5, 2.0]
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot convergence for each initial value
        for initial in initial_values:
            _, sequence, _ = self.derive_lz(initial, max_iterations)
            iterations = list(range(len(sequence)))
            
            ax.plot(iterations, sequence, 'o-', label=f'Start: {initial}')
            
        # Add horizontal line for LZ
        ax.axhline(y=self.LZ, color='r', linestyle='--', label=f'LZ = {self.LZ:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Convergence to LZ')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def stability_analysis(self, x_min: float = 0, 
                         x_max: float = 3, 
                         points: int = 1000) -> plt.Figure:
        """
        Plot stability analysis of the recursive function.
        
        Args:
            x_min: Minimum x value
            x_max: Maximum x value
            points: Number of points to plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate x values
        x = np.linspace(x_min, x_max, points)
        
        # Calculate stability at each point
        stability = np.array([self.stability_at_point(xi) for xi in x])
        
        # Plot stability
        ax.plot(x, stability, 'b-', label='|f\'(x)|')
        
        # Add horizontal line at y=1
        ax.axhline(y=1, color='r', linestyle='--', label='Stability threshold')
        
        # Mark LZ
        lz_stability = self.stability_at_point(self.LZ)
        ax.plot([self.LZ], [lz_stability], 'go', markersize=8, 
                label=f'LZ stability: {lz_stability:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('x')
        ax.set_ylabel('|f\'(x)|')
        ax.set_title('Stability Analysis')
        ax.legend()
        ax.grid(True)
        
        return fig
    
    def hqs_threshold_function(self, phase_diff: float) -> float:
        """
        Apply HQS threshold to phase difference.
        
        Args:
            phase_diff: Phase difference value
            
        Returns:
            0 if below threshold, 1 if above
        """
        threshold = 2 * np.pi * self.HQS
        return 0 if phase_diff < threshold else 1
    
    def recursive_hqs(self, phase_diffs: List[float], 
                    depth: int = 5) -> List[float]:
        """
        Apply recursive HQS threshold to phase differences.
        
        Args:
            phase_diffs: List of phase differences
            depth: Recursion depth
            
        Returns:
            List of HQS values after recursion
        """
        results = []
        
        for phase in phase_diffs:
            # Apply initial threshold
            current = self.hqs_threshold_function(phase)
            
            # Apply recursive thresholds
            for _ in range(depth):
                current = self.hqs_threshold_function(current * np.pi)
                
            results.append(current)
            
        return results


class OctaveModule:
    """
    Module for octave-related calculations and transformations.
    
    This module provides functions for working with octave reductions,
    transformations, and related operations.
    """
    
    def __init__(self, lz_module: LZModule = None):
        """
        Initialize the Octave module.
        
        Args:
            lz_module: Reference to LZ module
        """
        self.lz_module = lz_module if lz_module else LZModule()
        
    def octave_reduction(self, n: int) -> int:
        """
        Reduce a number to its octave (1-9).
        
        Args:
            n: Number to reduce
            
        Returns:
            Octave value (1-9)
        """
        if n == 0:
            return 9
            
        # Sum digits until single digit
        while n > 9:
            n = sum(int(digit) for digit in str(n))
            
        return n
    
    def octave_reduction_sequence(self, sequence: List[int]) -> List[int]:
        """
        Apply octave reduction to a sequence of numbers.
        
        Args:
            sequence: List of numbers
            
        Returns:
            List of octave values
        """
        return [self.octave_reduction(n) for n in sequence]
    
    def lz_based_octave(self, value: float) -> float:
        """
        Map a value to [0,1) based on LZ scaling.
        
        Args:
            value: Value to map
            
        Returns:
            Mapped value in [0,1)
        """
        # Find the power of LZ that brings value into [1,LZ)
        if value <= 0:
            return 0  # Handle non-positive values
            
        power = math.floor(math.log(value) / math.log(self.lz_module.LZ))
        
        # Scale to [1,LZ)
        scaled = value / (self.lz_module.LZ ** power)
        
        # Map to [0,1)
        mapped = (scaled - 1) / (self.lz_module.LZ - 1)
        
        return mapped
    
    def collatz_octave_transform(self, n: int, key: int, 
                               steps: int) -> List[int]:
        """
        Apply Collatz-like transformation and convert to octaves.
        
        Args:
            n: Starting number
            key: Transformation key
            steps: Maximum number of steps
            
        Returns:
            List of octave values
        """
        sequence = []
        current = n
        
        for _ in range(steps):
            # Apply octave reduction
            octave = self.octave_reduction(current)
            sequence.append(octave)
            
            # Apply Collatz-like transformation
            if current % 2 == 0:
                current = current // 2
            else:
                current = 3 * current + key
                
            # Check for cycle to 1,4,2,1,...
            if current == 1 and len(sequence) > 3:
                if sequence[-3:] == [4, 2, 1]:
                    break
                    
        return sequence
    
    def octave_distribution(self, sequence: List[int]) -> Dict[int, int]:
        """
        Calculate distribution of octaves in a sequence.
        
        Args:
            sequence: Sequence of numbers
            
        Returns:
            Dictionary mapping octave to count
        """
        # Convert to octaves
        octaves = self.octave_reduction_sequence(sequence)
        
        # Count occurrences
        distribution = {}
        for i in range(1, 10):
            distribution[i] = octaves.count(i)
            
        return distribution
    
    def plot_octave_distribution(self, sequence: List[int]) -> plt.Figure:
        """
        Plot distribution of octaves in a sequence.
        
        Args:
            sequence: Sequence of numbers
            
        Returns:
            Matplotlib figure
        """
        # Get distribution
        distribution = self.octave_distribution(sequence)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distribution
        octaves = list(distribution.keys())
        counts = list(distribution.values())
        
        ax.bar(octaves, counts)
        
        # Add labels
        ax.set_xlabel('Octave')
        ax.set_ylabel('Count')
        ax.set_title('Octave Distribution')
        ax.set_xticks(range(1, 10))
        ax.grid(True, axis='y')
        
        return fig
