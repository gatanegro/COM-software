"""
COM Framework Core Module - Fixed Version

This module implements the core functionality of the Continuous Oscillatory
Model (COM) framework, including the LZ constant derivation and octave mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
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
        # LZ constant (derived from recursive wave function)
        self.LZ = 1.23498
        
        # HQS (Harmonic Quantum Scalar) threshold (23.5% of LZ)
        self.HQS = 0.235 * self.LZ
        
    def recursive_wave_function(self, x: float) -> float:
        """
        Calculate the recursive wave function value at a point.
        
        Args:
            x: Input value
            
        Returns:
            Function value f(x) = sin(x) + e^(-x)
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
            Tuple of (derived LZ value, sequence of iterations, iteration count)
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
            
        # Return the final value if max iterations reached
        return current, sequence, max_iterations
    
    def verify_lz(self) -> bool:
        """
        Verify that LZ is a fixed point of the recursive function.
        
        Returns:
            True if LZ is a fixed point, False otherwise
        """
        # Calculate function value at LZ
        f_lz = self.recursive_wave_function(self.LZ)
        
        # Check if it equals LZ (within precision)
        return abs(f_lz - self.LZ) < 1e-5
    
    def stability_at_point(self, x: float) -> float:
        """
        Calculate the stability of the recursive function at a point.
        
        Args:
            x: Point to evaluate stability at
            
        Returns:
            Absolute value of the derivative at x
        """
        # Derivative of sin(x) + e^(-x) is cos(x) - e^(-x)
        derivative = np.cos(x) - np.exp(-x)
        
        # Return absolute value (magnitude)
        return abs(derivative)
    
    def is_stable_fixed_point(self, x: float) -> bool:
        """
        Check if a point is a stable fixed point.
        
        Args:
            x: Point to check
            
        Returns:
            True if x is a stable fixed point, False otherwise
        """
        # Check if it's a fixed point
        is_fixed_point = abs(self.recursive_wave_function(x) - x) < 1e-5
        
        # Check stability (derivative magnitude < 1)
        is_stable = self.stability_at_point(x) < 1
        
        return bool(is_fixed_point and is_stable)
    
    def lz_scaling(self, value: float, octave: int) -> float:
        """
        Scale a value by a power of LZ.
        
        Args:
            value: Base value
            octave: Power to scale by
            
        Returns:
            Scaled value (value * LZ^octave)
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
        
        # Check points in the range
        x = start
        while x <= end:
            # Calculate function value
            f_x = self.recursive_wave_function(x)
            
            # Check if it's a fixed point
            if abs(f_x - x) < step:
                fixed_points.append(x)
                
            x += step
            
        return fixed_points
    
    def plot_recursive_function(self) -> plt.Figure:
        """
        Plot the recursive function and identity line.
        
        Returns:
            Matplotlib figure with the plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate x values
        x = np.linspace(0, 3, 1000)
        
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
            Matplotlib figure with the plot
        """
        if initial_values is None:
            initial_values = [0.5, 1.0, 1.5, 2.0]
            
        # Create figure
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
    
    def stability_analysis(self) -> plt.Figure:
        """
        Plot stability analysis of the recursive function.
        
        Returns:
            Matplotlib figure with the plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate x values
        x = np.linspace(0, 3, 1000)
        
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
        Apply the HQS threshold function to a phase difference.
        
        Args:
            phase_diff: Phase difference value
            
        Returns:
            0 if below threshold, 1 if above
        """
        threshold = 2 * np.pi * self.HQS
        
        if phase_diff < threshold:
            return 0
        else:
            return 1
    
    def recursive_hqs(self, phase_diffs: List[float], 
                    depth: int = 3) -> List[float]:
        """
        Apply recursive HQS threshold function to a list of phase differences.
        
        Args:
            phase_diffs: List of phase differences
            depth: Recursion depth
            
        Returns:
            List of resulting values
        """
        results = [self.hqs_threshold_function(pd) for pd in phase_diffs]
        
        # Apply recursively
        for _ in range(1, depth):
            # Calculate new phase differences
            new_phase_diffs = [pd * self.LZ for pd in phase_diffs]
            
            # Apply threshold function
            new_results = [self.hqs_threshold_function(pd) for pd in new_phase_diffs]
            
            # Combine results
            results = [r1 * r2 for r1, r2 in zip(results, new_results)]
            
            # Update phase differences for next iteration
            phase_diffs = new_phase_diffs
            
        return results


class OctaveModule:
    """
    Module for octave-related calculations and transformations.
    
    This module provides functions for working with octave reductions,
    distributions, and transformations based on the COM framework.
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
            return 9  # Special case: 0 maps to 9
            
        # Calculate digital root
        while n > 9:
            n = sum(int(digit) for digit in str(n))
            
        return n
    
    def octave_reduction_sequence(self, sequence: List[int]) -> List[int]:
        """
        Apply octave reduction to a sequence of numbers.
        
        Args:
            sequence: List of numbers to reduce
            
        Returns:
            List of octave values
        """
        return [self.octave_reduction(n) for n in sequence]
    
    def lz_based_octave(self, value: float) -> float:
        """
        Map a value to an octave position using LZ-based scaling.
        
        Args:
            value: Value to map
            
        Returns:
            Octave position (0-1)
        """
        # Take log base LZ
        if value <= 0:
            return 0
            
        log_val = np.log(value) / np.log(self.lz_module.LZ)
        
        # Take fractional part
        frac_part = log_val - np.floor(log_val)
        
        # Ensure result is in [0, 1)
        if frac_part >= 0.999999:  # Handle floating point precision issues
            return 0.0
        
        return frac_part
    
    def collatz_octave_transform(self, n: int, key: int, 
                               steps: int) -> List[int]:
        """
        Apply Collatz-Octave transformation to a number.
        
        Args:
            n: Starting number
            key: Transformation key
            steps: Maximum number of steps
            
        Returns:
            List of octave values in the transformation
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
                
            # Stop if we reach 1
            if current == 1:
                sequence.append(1)
                break
                
        return sequence
    
    def octave_distribution(self, sequence: List[int]) -> Dict[int, int]:
        """
        Calculate the distribution of octaves in a sequence.
        
        Args:
            sequence: Sequence of numbers
            
        Returns:
            Dictionary mapping octaves to counts
        """
        # Apply octave reduction
        octaves = self.octave_reduction_sequence(sequence)
        
        # Count occurrences
        distribution = {i: octaves.count(i) for i in range(1, 10)}
        
        return distribution
    
    def plot_octave_distribution(self, sequence: List[int]) -> plt.Figure:
        """
        Plot the distribution of octaves in a sequence.
        
        Args:
            sequence: Sequence of numbers
            
        Returns:
            Matplotlib figure with the plot
        """
        # Calculate distribution
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
