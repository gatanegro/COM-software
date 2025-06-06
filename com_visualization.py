"""
COM Framework Visualization Module Fixes

This module implements fixes for the visualization components of the Continuous Oscillatory
Model (COM) framework, addressing issues identified in testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union, Callable
import math

# Import the core modules
from com_framework_core_fixed import LZModule, OctaveModule

class VisualizationModule:
    """
    Visualization tools for the COM framework.
    
    This module provides functions for creating visualizations of
    LZ-based patterns, octave distributions, and other COM concepts.
    """
    
    def __init__(self, lz_module: LZModule = None, octave_module: OctaveModule = None):
        """
        Initialize the Visualization module.
        
        Args:
            lz_module: Reference to LZ module
            octave_module: Reference to Octave module
        """
        self.lz_module = lz_module if lz_module else LZModule()
        self.octave_module = octave_module if octave_module else OctaveModule(self.lz_module)
        
        # Set up color schemes
        self.colors = sns.color_palette("viridis", 9)
        
        # Configure plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        
    def create_dashboard(self) -> plt.Figure:
        """
        Create a dashboard with multiple visualizations.
        
        Returns:
            Matplotlib figure with dashboard
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Add recursive function plot
        ax1 = fig.add_subplot(2, 3, 1)
        self._plot_recursive_function(ax1)
        
        # Add convergence plot
        ax2 = fig.add_subplot(2, 3, 2)
        self._plot_convergence(ax2)
        
        # Add stability analysis
        ax3 = fig.add_subplot(2, 3, 3)
        self._plot_stability_analysis(ax3)
        
        # Add octave distribution
        ax4 = fig.add_subplot(2, 3, 4)
        self._plot_octave_distribution(ax4)
        
        # Add LZ powers visualization
        ax5 = fig.add_subplot(2, 3, 5)
        self._plot_lz_powers(ax5)
        
        # Add HQS threshold visualization
        ax6 = fig.add_subplot(2, 3, 6)
        self._plot_hqs_threshold(ax6)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _plot_recursive_function(self, ax: plt.Axes) -> None:
        """
        Plot the recursive function on a given axis.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # Generate x values
        x = np.linspace(0, 3, 1000)
        
        # Calculate function values
        y = np.array([self.lz_module.recursive_wave_function(xi) for xi in x])
        
        # Plot function
        ax.plot(x, y, 'b-', label='sin(x) + e^(-x)')
        
        # Plot identity line
        ax.plot(x, x, 'r--', label='y = x')
        
        # Mark LZ
        ax.plot([self.lz_module.LZ], [self.lz_module.LZ], 'go', markersize=8, label=f'LZ = {self.lz_module.LZ:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Recursive Wave Function')
        ax.legend()
        ax.grid(True)
    
    def _plot_convergence(self, ax: plt.Axes) -> None:
        """
        Plot convergence to LZ on a given axis.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # Initial values
        initial_values = [0.5, 1.0, 1.5, 2.0]
        max_iterations = 20
        
        # Plot convergence for each initial value
        for initial in initial_values:
            _, sequence, _ = self.lz_module.derive_lz(initial, max_iterations)
            iterations = list(range(len(sequence)))
            
            ax.plot(iterations, sequence, 'o-', label=f'Start: {initial}')
            
        # Add horizontal line for LZ
        ax.axhline(y=self.lz_module.LZ, color='r', linestyle='--', label=f'LZ = {self.lz_module.LZ:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Convergence to LZ')
        ax.legend()
        ax.grid(True)
    
    def _plot_stability_analysis(self, ax: plt.Axes) -> None:
        """
        Plot stability analysis on a given axis.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # Generate x values
        x = np.linspace(0, 3, 1000)
        
        # Calculate stability at each point
        stability = np.array([self.lz_module.stability_at_point(xi) for xi in x])
        
        # Plot stability
        ax.plot(x, stability, 'b-', label='|f\'(x)|')
        
        # Add horizontal line at y=1
        ax.axhline(y=1, color='r', linestyle='--', label='Stability threshold')
        
        # Mark LZ
        lz_stability = self.lz_module.stability_at_point(self.lz_module.LZ)
        ax.plot([self.lz_module.LZ], [lz_stability], 'go', markersize=8, 
                label=f'LZ stability: {lz_stability:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('x')
        ax.set_ylabel('|f\'(x)|')
        ax.set_title('Stability Analysis')
        ax.legend()
        ax.grid(True)
    
    def _plot_octave_distribution(self, ax: plt.Axes) -> None:
        """
        Plot octave distribution on a given axis.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # Generate sequence
        sequence = list(range(1, 1001))
        
        # Get distribution
        distribution = self.octave_module.octave_distribution(sequence)
        
        # Plot distribution
        octaves = list(distribution.keys())
        counts = list(distribution.values())
        
        ax.bar(octaves, counts, color=self.colors)
        
        # Add labels
        ax.set_xlabel('Octave')
        ax.set_ylabel('Count')
        ax.set_title('Octave Distribution (1-1000)')
        ax.set_xticks(range(1, 10))
        ax.grid(True, axis='y')
    
    def _plot_lz_powers(self, ax: plt.Axes) -> None:
        """
        Plot LZ powers on a given axis.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # Generate powers
        powers = range(-5, 6)
        values = [self.lz_module.LZ ** p for p in powers]
        
        # Plot values
        ax.plot(powers, values, 'o-', color='blue')
        
        # Add labels
        ax.set_xlabel('Power')
        ax.set_ylabel('LZ^power')
        ax.set_title('Powers of LZ')
        ax.grid(True)
        
        # Add text annotations
        for p, v in zip(powers, values):
            ax.annotate(f'{v:.5f}', (p, v), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
    
    def _plot_hqs_threshold(self, ax: plt.Axes) -> None:
        """
        Plot HQS threshold on a given axis.
        
        Args:
            ax: Matplotlib axis to plot on
        """
        # Generate phase differences
        phase_diffs = np.linspace(0, 2 * np.pi, 1000)
        
        # Apply threshold function
        thresholds = [self.lz_module.hqs_threshold_function(pd) for pd in phase_diffs]
        
        # Plot threshold function
        ax.plot(phase_diffs, thresholds, 'b-')
        
        # Mark HQS threshold
        hqs_threshold = 2 * np.pi * self.lz_module.HQS
        ax.axvline(x=hqs_threshold, color='r', linestyle='--', 
                   label=f'HQS threshold: {hqs_threshold:.5f}')
        
        # Add labels and legend
        ax.set_xlabel('Phase difference')
        ax.set_ylabel('Threshold output')
        ax.set_title('HQS Threshold Function')
        ax.legend()
        ax.grid(True)
    
    def create_energy_pattern_visualization(self, size: int = 100, 
                                          iterations: int = 10) -> plt.Figure:
        """
        Create a visualization of energy patterns based on LZ.
        
        Args:
            size: Size of the grid
            iterations: Number of iterations
            
        Returns:
            Matplotlib figure with visualization
        """
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        
        # Create grid of values
        x = np.linspace(0, 2 * np.pi, size)
        y = np.linspace(0, 2 * np.pi, size)
        X, Y = np.meshgrid(x, y)
        
        # Initialize grid with sin(X) + sin(Y)
        Z = np.sin(X) + np.sin(Y)
        
        # Apply iterations of the recursive function
        for _ in range(iterations):
            Z = np.sin(Z) + np.exp(-np.abs(Z))
        
        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Energy')
        ax.set_title(f'Energy Pattern after {iterations} iterations')
        
        return fig
    
    def create_cryptographic_visualization(self, text: str) -> plt.Figure:
        """
        Create a visualization of cryptographic transformation.
        
        Args:
            text: Text to visualize
            
        Returns:
            Matplotlib figure with visualization
        """
        # Convert text to ASCII values
        ascii_values = [ord(c) for c in text]
        
        # Apply octave reduction
        octaves = [self.octave_module.octave_reduction(val) for val in ascii_values]
        
        # Apply Collatz-Octave transformation with different keys
        keys = [1, 3, 7, 11]
        transformations = []
        
        for key in keys:
            transformed = []
            for val in ascii_values:
                sequence = self.octave_module.collatz_octave_transform(val, key, 10)
                transformed.append(sequence)
            transformations.append(transformed)
        
        # Create figure
        fig, axes = plt.subplots(len(keys) + 1, 1, figsize=(12, 10), sharex=True)
        
        # Plot original values
        axes[0].stem(range(len(ascii_values)), ascii_values, linefmt='b-', markerfmt='bo')
        axes[0].set_ylabel('ASCII')
        axes[0].set_title('Original Text (ASCII)')
        
        # Plot transformations
        for i, (key, transformed) in enumerate(zip(keys, transformations)):
            # Flatten the transformation for visualization
            flat = [item for sublist in transformed for item in sublist]
            
            # Plot
            axes[i+1].stem(range(len(flat)), flat, linefmt='g-', markerfmt='go')
            axes[i+1].set_ylabel('Octave')
            axes[i+1].set_title(f'Transformation (Key={key})')
        
        # Add labels
        axes[-1].set_xlabel('Position')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_interactive_lz_explorer(self) -> plt.Figure:
        """
        Create an interactive visualization for exploring LZ properties.
        
        Returns:
            Matplotlib figure with visualization
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Flatten axes for easier access
        axes = axes.flatten()
        
        # Plot recursive function
        self._plot_recursive_function(axes[0])
        
        # Plot stability analysis
        self._plot_stability_analysis(axes[1])
        
        # Plot LZ powers
        self._plot_lz_powers(axes[2])
        
        # Plot HQS threshold
        self._plot_hqs_threshold(axes[3])
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def create_lz_convergence_animation(self, initial_values: List[float] = None, 
                                      max_iterations: int = 20) -> animation.FuncAnimation:
        """
        Create an animation of convergence to LZ.
        
        Args:
            initial_values: List of starting values
            max_iterations: Maximum number of iterations
            
        Returns:
            Matplotlib animation
        """
        if initial_values is None:
            initial_values = [0.5, 1.0, 1.5, 2.0]
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate sequences
        sequences = []
        for initial in initial_values:
            _, sequence, _ = self.lz_module.derive_lz(initial, max_iterations)
            sequences.append(sequence)
        
        # Find y-axis limits
        y_min = min(min(seq) for seq in sequences) - 0.1
        y_max = max(max(seq) for seq in sequences) + 0.1
        
        # Set up plot
        ax.set_xlim(0, max_iterations)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Convergence to LZ')
        ax.grid(True)
        
        # Add horizontal line for LZ
        ax.axhline(y=self.lz_module.LZ, color='r', linestyle='--', label=f'LZ = {self.lz_module.LZ:.5f}')
        
        # Create line objects
        lines = []
        for i, initial in enumerate(initial_values):
            line, = ax.plot([], [], 'o-', label=f'Start: {initial}')
            lines.append(line)
            
        # Add legend
        ax.legend()
        
        # Animation update function
        def update(frame):
            for i, line in enumerate(lines):
                if frame < len(sequences[i]):
                    x = list(range(frame + 1))
                    y = sequences[i][:frame + 1]
                    line.set_data(x, y)
            return lines
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=max_iterations + 1, 
                                       interval=200, blit=True)
        
        return anim
    
    def save_all_visualizations(self, output_dir: str) -> Dict[str, str]:
        """
        Save all visualizations to files.
        
        Args:
            output_dir: Directory to save files
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store file paths
        file_paths = {}
        
        # Save dashboard
        dashboard_path = os.path.join(output_dir, 'dashboard.png')
        dashboard = self.create_dashboard()
        dashboard.savefig(dashboard_path)
        plt.close(dashboard)
        file_paths['dashboard'] = dashboard_path
        
        # Save energy pattern visualization
        energy_path = os.path.join(output_dir, 'energy_pattern.png')
        energy = self.create_energy_pattern_visualization()
        energy.savefig(energy_path)
        plt.close(energy)
        file_paths['energy_pattern'] = energy_path
        
        # Save cryptographic visualization
        crypto_path = os.path.join(output_dir, 'cryptographic.png')
        crypto = self.create_cryptographic_visualization('COM Framework')
        crypto.savefig(crypto_path)
        plt.close(crypto)
        file_paths['cryptographic'] = crypto_path
        
        # Save LZ explorer
        explorer_path = os.path.join(output_dir, 'lz_explorer.png')
        explorer = self.create_interactive_lz_explorer()
        explorer.savefig(explorer_path)
        plt.close(explorer)
        file_paths['lz_explorer'] = explorer_path
        
        # Save convergence animation frames
        anim = self.create_lz_convergence_animation()
        anim_path = os.path.join(output_dir, 'convergence_animation.gif')
        
        # Save animation as GIF
        try:
            anim.save(anim_path, writer='pillow', fps=5)
            file_paths['convergence_animation'] = anim_path
        except Exception as e:
            # If animation saving fails, save a static image instead
            static_path = os.path.join(output_dir, 'convergence_static.png')
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
            self._plot_convergence(ax)
            fig.savefig(static_path)
            plt.close(fig)
            file_paths['convergence_static'] = static_path
        
        plt.close(anim._fig)
        
        return file_paths