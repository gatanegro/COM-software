"""
COM Framework - Main Application

This is the main application file for the Collatz Octave Model (COM) framework.
It integrates all modules and provides a user interface for exploring the framework.
"""
import matplotlib
print(matplotlib.get_backend())  # See what backend is currently used
matplotlib.use('TkAgg')  # Or 'Qt5Agg', or another interactive backend
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox, 
    QDoubleSpinBox, QTextEdit, QFileDialog, QMessageBox, QGroupBox,
    QFormLayout, QLineEdit, QCheckBox, QRadioButton, QButtonGroup,
    QSlider, QProgressBar,QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
matplotlib.use('Qt5Agg')

# Import COM framework modules
from com_framework_core_fixed2 import LZModule, OctaveModule
from com_visualization_fixed import VisualizationModule
from com_analysis_fixed import (
    MathematicalAnalysisModule,
    PatternRecognitionModule,
    StatisticalAnalysisModule
)

class MatplotlibCanvas(FigureCanvas):
    """Canvas for Matplotlib figures in Qt."""
    
    def __init__(self, fig=None, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the canvas.
        
        Args:
            fig: Matplotlib figure (optional)
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        if fig is None:
            fig = plt.figure(figsize=(width, height), dpi=dpi)
            
        self.fig = fig
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)
        
        # Set up figure
        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)


class AnalysisWorker(QThread):
    """Worker thread for running analyses."""
    
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, analysis_func, args=None, kwargs=None):
        """
        Initialize the worker.
        
        Args:
            analysis_func: Function to run
            args: Positional arguments
            kwargs: Keyword arguments
        """
        super(AnalysisWorker, self).__init__()
        self.analysis_func = analysis_func
        self.args = args or []
        self.kwargs = kwargs or {}
        
    def run(self):
        """Run the analysis function."""
        try:
            # Run the analysis
            result = self.analysis_func(*self.args, **self.kwargs)
            
            # Emit finished signal with result
            self.finished.emit(result)
        except Exception as e:
            # Emit error signal
            self.error.emit(str(e))


class COMFrameworkApp(QMainWindow):
    """Main application window for the COM Framework."""
    
    def __init__(self):
        """Initialize the application."""
        super(COMFrameworkApp,self).__init__()
        
        # Initialize COM modules
        self.lz_module = LZModule()
        self.octave_module = OctaveModule(self.lz_module)
        self.viz_module = VisualizationModule(self.lz_module, self.octave_module)
        self.math_module = MathematicalAnalysisModule(self.lz_module, self.octave_module)
        self.pattern_module = PatternRecognitionModule(self.lz_module, self.octave_module)
        self.stats_module = StatisticalAnalysisModule(self.lz_module, self.octave_module)
        
        # Set up the UI
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Set window properties
        self.setWindowTitle('COM Framework Explorer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create header
        header_layout = QHBoxLayout()
        header_label = QLabel('Collatz Octave Model (COM) Framework')
        header_label.setFont(QFont('Arial', 16, QFont.Bold))
        header_layout.addWidget(header_label)
        main_layout.addLayout(header_layout)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.add_dashboard_tab()
        self.add_lz_explorer_tab()
        self.add_octave_analysis_tab()
        self.add_pattern_recognition_tab()
        self.add_visualization_tab()
        self.add_about_tab()
        
        # Create status bar
        self.statusBar().showMessage('Ready')
        
        # Show the window
        self.show()
        
    def add_dashboard_tab(self):
        """Add the dashboard tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
      
        # Add title
        title = QLabel('Welcome to the COM Framework Explorer')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
          # Add subtitle
        subtitle = QLabel('Educational & Research Software')
        subtitle.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(subtitle)
        
        # Add description
        description = QLabel(
           
         'COM Framework Software Architecture Design \n \n'
        
        'LZ Module \n'
        'Octave Module \n'
        'Energy Patterns Module \n'
        'HQS Module \n\n'
                      '•Applications Layer \n\n'
        'Cryptography Module \n'
        'Mathematical Analysis Module \n'
        'Pattern Recognition Module \n\n'
                     
         'Static Visualization Module \n'
         'Interactive Visualization Module \n'
                      '•User Interface Layer \n\n'
         'Command Line Interface Module\n'
                      '•Utilities Layer \n\n'
        'Data Import/Export Module \n\n'
        'COM Framework Explorer v1.0\n'
            'Released: June © 2025 Martin Doina (dhelamay@protonmail.com \n')
      
        layout.addWidget(description)
        
        # Create dashboard visualization
        dashboard_canvas = MatplotlibCanvas(width=10, height=8)
        layout.addWidget(dashboard_canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(dashboard_canvas, tab)
        layout.addWidget(toolbar)
        
        # Create dashboard
        dashboard_fig = self.viz_module.create_dashboard()
        dashboard_canvas.fig = dashboard_fig
        dashboard_canvas.draw()
        
        # Add to tabs
        self.tabs.addTab(tab, 'Dashboard')   
        
    def add_lz_explorer_tab(self):
        """Add the LZ explorer tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add description
        description = QLabel(
            'Explore the properties of the LZ constant (1.23498) and related functions. '
            'This tab allows you to visualize the recursive wave function, stability analysis, '
            'and convergence properties.'
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Function selection
        function_group = QGroupBox('Function')
        function_layout = QVBoxLayout(function_group)
        
        self.function_combo = QComboBox()
        self.function_combo.addItems([
            'Recursive Wave Function',
            'Convergence to LZ',
            'Stability Analysis',
            'Fixed Points',
            'HQS Threshold'
        ])
        function_layout.addWidget(self.function_combo)
        
        # Parameters
        params_group = QGroupBox('Parameters')
        params_layout = QFormLayout(params_group)
        
        self.start_value = QDoubleSpinBox()
        self.start_value.setRange(0, 10)
        self.start_value.setValue(0)
        self.start_value.setSingleStep(0.1)
        params_layout.addRow('Start:', self.start_value)
        
        self.end_value = QDoubleSpinBox()
        self.end_value.setRange(0, 10)
        self.end_value.setValue(3)
        self.end_value.setSingleStep(0.1)
        params_layout.addRow('End:', self.end_value)
        
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 100)
        self.iterations.setValue(20)
        params_layout.addRow('Iterations:', self.iterations)
        
        # Add button
        self.plot_button = QPushButton('Plot')
        self.plot_button.clicked.connect(self.plot_lz_function)
        
        # Add controls to layout
        controls_layout.addWidget(function_group)
        controls_layout.addWidget(params_group)
        controls_layout.addWidget(self.plot_button)
        layout.addLayout(controls_layout)
        
        # Create plot area
        self.lz_canvas = MatplotlibCanvas(width=10, height=6)
        layout.addWidget(self.lz_canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(self.lz_canvas, tab)
        layout.addWidget(toolbar)
        
        # Add to tabs
        self.tabs.addTab(tab, 'LZ Explorer')
        
    def plot_lz_function(self):
        """Plot the selected LZ function."""
        function_index = self.function_combo.currentIndex()
        
        # Clear the canvas
        self.lz_canvas.fig.clear()
        ax = self.lz_canvas.fig.add_subplot(111)
        
        # Plot the selected function
        if function_index == 0:  # Recursive Wave Function
            # Generate x values
            x = np.linspace(self.start_value.value(), self.end_value.value(), 1000)
            
            # Calculate function values
            y = np.array([self.lz_module.recursive_wave_function(xi) for xi in x])
            
            # Plot function
            ax.plot(x, y, 'b-', label='sin(x) + e^(-x)')
            
            # Plot identity line
            ax.plot(x, x, 'r--', label='y = x')
            
            # Mark LZ
            ax.plot([self.lz_module.LZ], [self.lz_module.LZ], 'go', markersize=8, 
                    label=f'LZ = {self.lz_module.LZ:.5f}')
            
            # Add labels and legend
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title('Recursive Wave Function')
            ax.legend()
            ax.grid(True)
            
        elif function_index == 1:  # Convergence to LZ
            # Initial values
            initial_values = [0.5, 1.0, 1.5, 2.0]
            max_iterations = self.iterations.value()
            
            # Plot convergence for each initial value
            for initial in initial_values:
                _, sequence, _ = self.lz_module.derive_lz(initial, max_iterations)
                iterations = list(range(len(sequence)))
                
                ax.plot(iterations, sequence, 'o-', label=f'Start: {initial}')
                
            # Add horizontal line for LZ
            ax.axhline(y=self.lz_module.LZ, color='r', linestyle='--', 
                       label=f'LZ = {self.lz_module.LZ:.5f}')
            
            # Add labels and legend
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Value')
            ax.set_title('Convergence to LZ')
            ax.legend()
            ax.grid(True)
            
        elif function_index == 2:  # Stability Analysis
            # Generate x values
            x = np.linspace(self.start_value.value(), self.end_value.value(), 1000)
            
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
            
        elif function_index == 3:  # Fixed Points
            # Find fixed points
            fixed_points = self.lz_module.find_fixed_points(
                self.start_value.value(), 
                self.end_value.value(), 
                0.01
            )
            
            # Generate x values
            x = np.linspace(self.start_value.value(), self.end_value.value(), 1000)
            
            # Calculate function values
            y = np.array([self.lz_module.recursive_wave_function(xi) for xi in x])
            
            # Plot function
            ax.plot(x, y, 'b-', label='sin(x) + e^(-x)')
            
            # Plot identity line
            ax.plot(x, x, 'r--', label='y = x')
            
            # Mark fixed points
            for fp in fixed_points:
                stability = self.lz_module.stability_at_point(fp)
                color = 'g' if stability < 1 else 'r'
                marker = 'o' if stability < 1 else 'x'
                ax.plot([fp], [fp], color + marker, markersize=8)
                
            # Add labels and legend
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.set_title(f'Fixed Points ({len(fixed_points)} found)')
            ax.legend()
            ax.grid(True)
            
        elif function_index == 4:  # HQS Threshold
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
        
        # Update the canvas
        self.lz_canvas.draw()
        
    def add_octave_analysis_tab(self):
        """Add the octave analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add description
        description = QLabel(
            'Analyze octave patterns and distributions. This tab allows you to explore '
            'octave reductions, distributions, and transformations based on the COM framework.'
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Analysis type selection
        analysis_group = QGroupBox('Analysis Type')
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.octave_analysis_combo = QComboBox()
        self.octave_analysis_combo.addItems([
            'Octave Distribution',
            'Collatz-Octave Transform',
            'LZ-Based Octave Mapping',
            'Octave Randomness'
        ])
        analysis_layout.addWidget(self.octave_analysis_combo)
        
        # Parameters
        octave_params_group = QGroupBox('Parameters')
        octave_params_layout = QFormLayout(octave_params_group)
        
        self.start_number = QSpinBox()
        self.start_number.setRange(1, 10000)
        self.start_number.setValue(1)
        octave_params_layout.addRow('Start:', self.start_number)
        
        self.end_number = QSpinBox()
        self.end_number.setRange(1, 10000)
        self.end_number.setValue(100)
        octave_params_layout.addRow('End:', self.end_number)
        
        self.transform_key = QSpinBox()
        self.transform_key.setRange(1, 100)
        self.transform_key.setValue(7)
        octave_params_layout.addRow('Key:', self.transform_key)
        
        # Add button
        self.analyze_button = QPushButton('Analyze')
        self.analyze_button.clicked.connect(self.analyze_octaves)
        
        # Add controls to layout
        controls_layout.addWidget(analysis_group)
        controls_layout.addWidget(octave_params_group)
        controls_layout.addWidget(self.analyze_button)
        layout.addLayout(controls_layout)
        
        # Create plot area
        self.octave_canvas = MatplotlibCanvas(width=10, height=6)
        layout.addWidget(self.octave_canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(self.octave_canvas, tab)
        layout.addWidget(toolbar)
        
        # Create results area
        self.octave_results = QTextEdit()
        self.octave_results.setReadOnly(True)
        layout.addWidget(self.octave_results)
        
        # Add to tabs
        self.tabs.addTab(tab, 'Octave Analysis')
        
    def analyze_octaves(self):
        """Analyze octaves based on selected parameters."""
        analysis_index = self.octave_analysis_combo.currentIndex()
        
        # Clear the canvas and results
        self.octave_canvas.fig.clear()
        self.octave_results.clear()
        
        # Create progress bar in status bar
        self.statusBar().showMessage('Analyzing...')
        
        # Create worker thread
        if analysis_index == 0:  # Octave Distribution
            # Generate sequence
            sequence = list(range(self.start_number.value(), self.end_number.value() + 1))
            
            # Calculate distribution
            distribution = self.octave_module.octave_distribution(sequence)
            
            # Plot distribution
            ax = self.octave_canvas.fig.add_subplot(111)
            octaves = list(distribution.keys())
            counts = list(distribution.values())
            
            ax.bar(octaves, counts)
            
            # Add labels
            ax.set_xlabel('Octave')
            ax.set_ylabel('Count')
            ax.set_title('Octave Distribution')
            ax.set_xticks(range(1, 10))
            ax.grid(True, axis='y')
            
            # Update results
            results_text = "Octave Distribution Analysis\n"
            results_text += "==========================\n\n"
            results_text += f"Range: {self.start_number.value()} to {self.end_number.value()}\n"
            results_text += f"Total numbers: {len(sequence)}\n\n"
            results_text += "Distribution:\n"
            
            for octave, count in distribution.items():
                percentage = count / len(sequence) * 100
                results_text += f"  Octave {octave}: {count} ({percentage:.1f}%)\n"
                
            # Add statistical analysis
            results = self.math_module.analyze_octave_distribution(
                self.start_number.value(), 
                self.end_number.value()
            )
            
            results_text += "\nStatistical Analysis:\n"
            results_text += f"  Chi-square statistic: {results['chi2_stat']:.3f}\n"
            results_text += f"  p-value: {results['p_value']:.3f}\n"
            results_text += f"  Uniform distribution: {'Yes' if results['is_uniform'] else 'No'}\n"
            
            self.octave_results.setText(results_text)
            
        elif analysis_index == 1:  # Collatz-Octave Transform
            # Apply transformation
            n = self.start_number.value()
            key = self.transform_key.value()
            steps = 100
            
            sequence = self.octave_module.collatz_octave_transform(n, key, steps)
            
            # Plot sequence
            ax = self.octave_canvas.fig.add_subplot(111)
            ax.plot(range(len(sequence)), sequence, 'o-')
            
            # Add labels
            ax.set_xlabel('Step')
            ax.set_ylabel('Octave')
            ax.set_title(f'Collatz-Octave Transform (n={n}, key={key})')
            ax.set_yticks(range(1, 10))
            ax.grid(True)
            
            # Update results
            results_text = "Collatz-Octave Transform Analysis\n"
            results_text += "================================\n\n"
            results_text += f"Starting number: {n}\n"
            results_text += f"Key: {key}\n"
            results_text += f"Sequence length: {len(sequence)}\n\n"
            results_text += "Sequence:\n"
            results_text += str(sequence) + "\n\n"
            
            # Add pattern analysis
            patterns = self.pattern_module.detect_octave_patterns([n])
            
            results_text += "Pattern Analysis:\n"
            if patterns['patterns']:
                for i, pattern in enumerate(patterns['patterns']):
                    results_text += f"  Pattern {i+1}: {pattern['pattern']} "
                    results_text += f"(occurs {pattern['occurrences']} times)\n"
            else:
                results_text += "  No repeating patterns detected\n"
                
            # Add rhythm analysis
            rhythm = patterns['rhythm']
            results_text += "\nRhythm Analysis:\n"
            results_text += f"  Most common difference: {rhythm['most_common_diff'][0]} "
            results_text += f"(occurs {rhythm['most_common_diff'][1]} times)\n"
            results_text += f"  Alternating pattern: {'Yes' if rhythm['alternating_pattern'] else 'No'}\n"
            results_text += f"  Increasing: {'Yes' if rhythm['increasing'] else 'No'}\n"
            results_text += f"  Decreasing: {'Yes' if rhythm['decreasing'] else 'No'}\n"
            
            self.octave_results.setText(results_text)
            
        elif analysis_index == 2:  # LZ-Based Octave Mapping
            # Generate values
            values = np.logspace(
                np.log10(0.1), 
                np.log10(10), 
                1000
            )
            
            # Calculate octave positions
            octaves = [self.octave_module.lz_based_octave(v) for v in values]
            
            # Plot mapping
            ax = self.octave_canvas.fig.add_subplot(111)
            ax.semilogx(values, octaves, 'b.')
            
            # Add labels
            ax.set_xlabel('Value')
            ax.set_ylabel('Octave Position (0-1)')
            ax.set_title('LZ-Based Octave Mapping')
            ax.grid(True)
            
            # Mark powers of LZ
            for i in range(-3, 4):
                value = self.lz_module.LZ ** i
                ax.axvline(x=value, color='r', linestyle='--', alpha=0.5,
                           label=f'LZ^{i}' if i == 0 else None)
                
            ax.legend()
            
            # Update results
            results_text = "LZ-Based Octave Mapping Analysis\n"
            results_text += "===============================\n\n"
            results_text += f"LZ constant: {self.lz_module.LZ:.5f}\n\n"
            results_text += "Powers of LZ:\n"
            
            for i in range(-3, 4):
                value = self.lz_module.LZ ** i
                octave = self.octave_module.lz_based_octave(value)
                results_text += f"  LZ^{i}: {value:.5f} -> Octave position: {octave:.5f}\n"
                
            # Add scaling analysis
            results = self.math_module.analyze_octave_scaling()
            
            results_text += "\nOctave Scaling Analysis:\n"
            results_text += "  Ratios between consecutive octaves:\n"
            
            for i, ratio in enumerate(results['ratios']):
                results_text += f"    Octave {i} to {i+1}: {ratio:.5f}\n"
                
            results_text += f"\n  Pattern length: {results['pattern_length']}\n"
            
            self.octave_results.setText(results_text)
            
        elif analysis_index == 3:  # Octave Randomness
            # Generate sequence
            sequence = list(range(self.start_number.value(), self.end_number.value() + 1))
            
            # Apply octave reduction
            octaves = [self.octave_module.octave_reduction(n) for n in sequence]
            
            # Plot sequence
            ax = self.octave_canvas.fig.add_subplot(111)
            ax.plot(range(len(octaves)), octaves, 'b-')
            
            # Add labels
            ax.set_xlabel('Position')
            ax.set_ylabel('Octave')
            ax.set_title('Octave Sequence')
            ax.set_yticks(range(1, 10))
            ax.grid(True)
            
            # Analyze randomness
            results = self.stats_module.analyze_octave_randomness(sequence)
            
            # Update results
            results_text = "Octave Randomness Analysis\n"
            results_text += "=========================\n\n"
            results_text += f"Range: {self.start_number.value()} to {self.end_number.value()}\n"
            results_text += f"Total numbers: {len(sequence)}\n\n"
            
            results_text += "Runs Test:\n"
            results_text += f"  Statistic: {results['runs_test']['statistic']:.3f}\n"
            results_text += f"  p-value: {results['runs_test']['p_value']:.3f}\n"
            results_text += f"  Random: {'Yes' if results['runs_test']['is_random'] else 'No'}\n\n"
            
            results_text += "Chi-Square Test:\n"
            results_text += f"  Statistic: {results['chi_square_test']['statistic']:.3f}\n"
            results_text += f"  p-value: {results['chi_square_test']['p_value']:.3f}\n"
            results_text += f"  Uniform: {'Yes' if results['chi_square_test']['is_uniform'] else 'No'}\n\n"
            
            results_text += "Entropy:\n"
            results_text += f"  Value: {results['entropy']['value']:.3f}\n"
            results_text += f"  Normalized: {results['entropy']['normalized']:.3f}\n"
            results_text += f"  High entropy: {'Yes' if results['entropy']['is_high'] else 'No'}\n\n"
            
            results_text += "Overall Assessment:\n"
            results_text += f"  Random: {'Yes' if results['overall_assessment']['is_random'] else 'No'}\n"
            results_text += f"  Confidence: {results['overall_assessment']['confidence']:.3f}\n"
            
            self.octave_results.setText(results_text)
        
        # Update the canvas
        self.octave_canvas.draw()
        
        # Update status bar
        self.statusBar().showMessage('Analysis complete')
        
    def add_pattern_recognition_tab(self):
        """Add the pattern recognition tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add description
        description = QLabel(
            'Detect and analyze patterns using the COM framework. This tab provides tools '
            'for pattern recognition, clustering, and feature extraction based on octave patterns.'
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Pattern type selection
        pattern_group = QGroupBox('Pattern Type')
        pattern_layout = QVBoxLayout(pattern_group)
        
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems([
            'Octave Patterns',
            'LZ-Based Patterns',
            'Clustering',
            'Feature Extraction'
        ])
        pattern_layout.addWidget(self.pattern_combo)
        
        # Input group
        input_group = QGroupBox('Input')
        input_layout = QVBoxLayout(input_group)
        
        self.pattern_input = QTextEdit()
        self.pattern_input.setPlaceholderText('Enter numbers separated by commas or spaces')
        input_layout.addWidget(self.pattern_input)
        
        # Add sample data button
        sample_button = QPushButton('Load Sample Data')
        sample_button.clicked.connect(self.load_sample_pattern_data)
        input_layout.addWidget(sample_button)
        
        # Add analyze button
        self.pattern_button = QPushButton('Analyze Patterns')
        self.pattern_button.clicked.connect(self.analyze_patterns)
        
        # Add controls to layout
        controls_layout.addWidget(pattern_group)
        controls_layout.addWidget(input_group)
        controls_layout.addWidget(self.pattern_button)
        layout.addLayout(controls_layout)
        
        # Create plot area
        self.pattern_canvas = MatplotlibCanvas(width=10, height=6)
        layout.addWidget(self.pattern_canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(self.pattern_canvas, tab)
        layout.addWidget(toolbar)
        
        # Create results area
        self.pattern_results = QTextEdit()
        self.pattern_results.setReadOnly(True)
        layout.addWidget(self.pattern_results)
        
        # Add to tabs
        self.tabs.addTab(tab, 'Pattern Recognition')
        
    def load_sample_pattern_data(self):
        """Load sample data for pattern recognition."""
        pattern_index = self.pattern_combo.currentIndex()
        
        if pattern_index == 0:  # Octave Patterns
            # Fibonacci sequence
            sample_data = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        elif pattern_index == 1:  # LZ-Based Patterns
            # Sine wave with LZ frequency
            sample_data = [np.sin(i * self.lz_module.LZ) for i in range(100)]
        elif pattern_index == 2:  # Clustering
            # Multiple sequences
            sample_data = "1,2,3,4,5\n5,6,7,8,9\n1,3,5,7,9\n2,4,6,8,1\n1,2,3,4,5"
            self.pattern_input.setText(sample_data)
            return
        elif pattern_index == 3:  # Feature Extraction
            # Random sequence
            sample_data = list(range(1, 101))
            
        # Convert to string and set in input field
        self.pattern_input.setText(', '.join(str(x) for x in sample_data))
        
    def analyze_patterns(self):
        """Analyze patterns based on selected parameters."""
        pattern_index = self.pattern_combo.currentIndex()
        
        # Parse input data
        input_text = self.pattern_input.toPlainText().strip()
        
        # Clear the canvas and results
        self.pattern_canvas.fig.clear()
        self.pattern_results.clear()
        
        # Create progress bar in status bar
        self.statusBar().showMessage('Analyzing patterns...')
        
        try:
            if pattern_index == 0 or pattern_index == 3:  # Octave Patterns or Feature Extraction
                # Parse as single sequence
                if ',' in input_text:
                    data = [int(x.strip()) for x in input_text.split(',') if x.strip()]
                else:
                    data = [int(x.strip()) for x in input_text.split() if x.strip()]
                    
            elif pattern_index == 1:  # LZ-Based Patterns
                # Parse as floating-point values
                if ',' in input_text:
                    data = [float(x.strip()) for x in input_text.split(',') if x.strip()]
                else:
                    data = [float(x.strip()) for x in input_text.split() if x.strip()]
                    
            elif pattern_index == 2:  # Clustering
                # Parse as multiple sequences
                lines = input_text.strip().split('\n')
                data_points = []
                
                for line in lines:
                    if ',' in line:
                        sequence = [int(x.strip()) for x in line.split(',') if x.strip()]
                    else:
                        sequence = [int(x.strip()) for x in line.split() if x.strip()]
                        
                    if sequence:
                        data_points.append(sequence)
        except ValueError:
            self.statusBar().showMessage('Error: Invalid input data')
            self.pattern_results.setText('Error: Please enter valid numeric data')
            return
            
        # Analyze based on pattern type
        if pattern_index == 0:  # Octave Patterns
            # Detect patterns
            results = self.pattern_module.detect_octave_patterns(data)
            
            # Plot octave sequence
            ax = self.pattern_canvas.fig.add_subplot(111)
            ax.plot(range(len(results['octaves'])), results['octaves'], 'o-')
            
            # Add labels
            ax.set_xlabel('Position')
            ax.set_ylabel('Octave')
            ax.set_title('Octave Sequence')
            ax.set_yticks(range(1, 10))
            ax.grid(True)
            
            # Highlight patterns if found
            if results['patterns']:
                top_pattern = results['patterns'][0]
                pattern_length = top_pattern['length']
                
                for pos in top_pattern['positions']:
                    ax.axvspan(pos, pos + pattern_length - 1, alpha=0.2, color='green')
                    
            # Update results
            results_text = "Octave Pattern Analysis\n"
            results_text += "======================\n\n"
            results_text += f"Input sequence length: {len(data)}\n"
            results_text += f"Octave sequence: {results['octaves']}\n\n"
            
            results_text += "Detected Patterns:\n"
            if results['patterns']:
                for i, pattern in enumerate(results['patterns']):
                    results_text += f"  Pattern {i+1}: {pattern['pattern']} "
                    results_text += f"(length: {pattern['length']}, occurrences: {pattern['occurrences']})\n"
                    results_text += f"    Positions: {pattern['positions']}\n"
            else:
                results_text += "  No repeating patterns detected\n"
                
            results_text += "\nFrequency Distribution:\n"
            for octave, freq in results['frequency_distribution'].items():
                results_text += f"  Octave {octave}: {freq:.3f}\n"
                
            results_text += "\nRhythm Analysis:\n"
            rhythm = results['rhythm']
            results_text += f"  Most common difference: {rhythm['most_common_diff'][0]} "
            results_text += f"(occurs {rhythm['most_common_diff'][1]} times)\n"
            results_text += f"  Alternating pattern: {'Yes' if rhythm['alternating_pattern'] else 'No'}\n"
            results_text += f"  Increasing: {'Yes' if rhythm['increasing'] else 'No'}\n"
            results_text += f"  Decreasing: {'Yes' if rhythm['decreasing'] else 'No'}\n"
            
            self.pattern_results.setText(results_text)
            
        elif pattern_index == 1:  # LZ-Based Patterns
            # Detect patterns
            results = self.pattern_module.detect_lz_based_patterns(data)
            
            # Plot data and autocorrelation
            fig = self.pattern_canvas.fig
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            
            # Plot original data
            ax1.plot(range(len(data)), data, 'b-')
            ax1.set_xlabel('Position')
            ax1.set_ylabel('Value')
            ax1.set_title('Input Data')
            ax1.grid(True)
            
            # Plot autocorrelation
            ax2.plot(range(len(results['autocorrelation'])), results['autocorrelation'], 'r-')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Autocorrelation')
            ax2.set_title('Autocorrelation')
            ax2.grid(True)
            
            # Mark peaks
            for peak in results['peaks']:
                ax2.axvline(x=peak, color='g', linestyle='--', alpha=0.5)
                
            # Adjust layout
            fig.tight_layout()
            
            # Update results
            results_text = "LZ-Based Pattern Analysis\n"
            results_text += "========================\n\n"
            results_text += f"Input sequence length: {len(data)}\n\n"
            
            results_text += "Autocorrelation Peaks:\n"
            if results['peaks']:
                for i, peak in enumerate(results['peaks']):
                    results_text += f"  Peak {i+1}: Lag {peak + 1}\n"
            else:
                results_text += "  No significant peaks detected\n"
                
            results_text += "\nDominant Frequencies:\n"
            for i, (freq, mag) in enumerate(results['dominant_frequencies']):
                results_text += f"  Frequency {i+1}: {abs(freq):.5f} (magnitude: {mag:.3f})\n"
                
            results_text += "\nLZ-Related Periods:\n"
            if results['lz_related_periods']:
                for i, period in enumerate(results['lz_related_periods']):
                    results_text += f"  Period {i+1}: {period['period']} "
                    results_text += f"(expected: {period['expected']:.3f}, "
                    results_text += f"power: {period['power']}, "
                    results_text += f"error: {period['error']:.3f})\n"
            else:
                results_text += "  No LZ-related periods detected\n"
                
            self.pattern_results.setText(results_text)
            
        elif pattern_index == 2:  # Clustering

            # Perform clustering
            results = self.pattern_module.cluster_by_octave_pattern(data_points)

            # Plot clusters
            ax = self.pattern_canvas.fig.add_subplot(111)

            # Convert sequences to 2D points
            if len(data_points[0]) > 2:
                from sklearn.manifold import TSNE

                # Create feature matrix
                min_length = min(len(seq) for seq in data_points)
                features = np.array([seq[:min_length] for seq in data_points])

                # Auto-adjust perplexity
                n_samples = features.shape[0]
                if n_samples >= 2:
                    perplexity = min(30, n_samples - 1)  # Cap at 30 or (n_samples-1)
                    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                    points_2d = tsne.fit_transform(features)
                else:
                    # Fallback for tiny datasets
                    points_2d = np.array([[seq[0], seq[1]] for seq in data_points])
            else:
                # Use first two elements directly
                points_2d = np.array([[seq[0], seq[1]] for seq in data_points])

            # Plot points colored by cluster
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            for i, label in enumerate(results['labels']):
                color = colors[label % len(colors)]
                ax.scatter(points_2d[i, 0], points_2d[i, 1], c=color, label=f'Cluster {label + 1}')

            # Add labels
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_title(f'Cluster Analysis (k={results["optimal_clusters"]})')

            # Add legend if not too many clusters
            if results['optimal_clusters'] <= 7:
                ax.legend()

            # Update results
            results_text = "Clustering Analysis\n"
            results_text += "==================\n\n"
            results_text += f"Number of sequences: {len(data_points)}\n"
            results_text += f"Sequence length: {min(len(seq) for seq in data_points)}\n\n"

            results_text += f"Optimal number of clusters: {results['optimal_clusters']}\n"
            results_text += f"Silhouette score: {results['silhouette_score']:.3f}\n\n"

            results_text += "Cluster Assignments:\n"
            for i, label in enumerate(results['labels']):
                results_text += f"  Sequence {i+1}: Cluster {label + 1}\n"

            results_text += "\nCluster Centers:\n"
            for i, center in enumerate(results['centers']):
                results_text += f"  Cluster {i+1}: {[round(x, 2) for x in center]}\n"

            self.pattern_results.setText(results_text)
            
        elif pattern_index == 3:  # Feature Extraction
            # Extract features
            results = self.pattern_module.find_octave_based_features(data)
            
            # Plot octave distribution
            ax = self.pattern_canvas.fig.add_subplot(111)
            
            # Get distribution
            octaves = list(range(1, 10))
            frequencies = [results['frequency_distribution'][i] for i in octaves]
            
            # Plot as bar chart
            ax.bar(octaves, frequencies)
            
            # Add labels
            ax.set_xlabel('Octave')
            ax.set_ylabel('Frequency')
            ax.set_title('Octave Distribution')
            ax.set_xticks(range(1, 10))
            ax.grid(True, axis='y')
            
            # Update results
            results_text = "Feature Extraction Analysis\n"
            results_text += "==========================\n\n"
            results_text += f"Input sequence length: {len(data)}\n\n"
            
            results_text += "Basic Statistics:\n"
            results_text += f"  Mean: {results['mean']:.3f}\n"
            results_text += f"  Median: {results['median']:.3f}\n"
            results_text += f"  Mode: {results['mode']:.3f}\n"
            results_text += f"  Standard Deviation: {results['std_dev']:.3f}\n"
            results_text += f"  Entropy: {results['entropy']:.3f}\n\n"
            
            results_text += "Run Length Analysis:\n"
            runs = results['runs']
            results_text += f"  Maximum run: {runs['max_run']}\n"
            results_text += f"  Average run: {runs['avg_run']:.3f}\n"
            results_text += "  Run counts:\n"
            for length, count in sorted(runs['run_counts'].items()):
                results_text += f"    Length {length}: {count}\n"
                
            results_text += "\nTransition Analysis:\n"
            transitions = results['transitions']
            results_text += f"  Direction changes: {transitions['direction_changes']}\n"
            results_text += f"  Average step: {transitions['avg_step']:.3f}\n"
            
            self.pattern_results.setText(results_text)
        
        # Update the canvas
        self.pattern_canvas.draw()
        
        # Update status bar
        self.statusBar().showMessage('Pattern analysis complete')
        
    def add_visualization_tab(self):
        """Add the visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add description
        description = QLabel(
            'Create visualizations based on the COM framework. This tab provides tools '
            'for generating various visualizations of LZ-based patterns, energy distributions, '
            'and other COM concepts.'
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Create controls
        controls_layout = QHBoxLayout()
        
        # Visualization type selection
        viz_group = QGroupBox('Visualization Type')
        viz_layout = QVBoxLayout(viz_group)
        
        self.viz_combo = QComboBox()
        self.viz_combo.addItems([
            'Energy Pattern',
            'Cryptographic Visualization',
            'LZ Convergence Animation',
            'Octave Mapping'
        ])
        viz_layout.addWidget(self.viz_combo)
        
        # Parameters
        viz_params_group = QGroupBox('Parameters')
        viz_params_layout = QFormLayout(viz_params_group)
        
        self.viz_size = QSpinBox()
        self.viz_size.setRange(50, 500)
        self.viz_size.setValue(100)
        self.viz_size.setSingleStep(10)
        viz_params_layout.addRow('Size:', self.viz_size)
        
        self.viz_iterations = QSpinBox()
        self.viz_iterations.setRange(1, 50)
        self.viz_iterations.setValue(10)
        viz_params_layout.addRow('Iterations:', self.viz_iterations)
        
        self.viz_text = QLineEdit()
        self.viz_text.setText('COM Framework')
        viz_params_layout.addRow('Text:', self.viz_text)
        
        # Add button
        self.viz_button = QPushButton('Generate Visualization')
        self.viz_button.clicked.connect(self.generate_visualization)
        
        # Add save button
        self.save_viz_button = QPushButton('Save Visualization')
        self.save_viz_button.clicked.connect(self.save_visualization)
        
        # Add controls to layout
        controls_layout.addWidget(viz_group)
        controls_layout.addWidget(viz_params_group)
        controls_layout.addWidget(self.viz_button)
        controls_layout.addWidget(self.save_viz_button)
        layout.addLayout(controls_layout)
        
        # Create visualization area
        self.viz_canvas = MatplotlibCanvas(width=10, height=8)
        layout.addWidget(self.viz_canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar(self.viz_canvas, tab)
        layout.addWidget(toolbar)
        
        # Add to tabs
        self.tabs.addTab(tab, 'Visualization')
        
    def generate_visualization(self):
        """Generate visualization based on selected parameters."""
        viz_index = self.viz_combo.currentIndex()
        
        # Clear the canvas
        plt.close(self.viz_canvas.fig)
        
        # Create progress bar in status bar
        self.statusBar().showMessage('Generating visualization...')
        
        # Generate visualization
        if viz_index == 0:  # Energy Pattern
            self.viz_canvas.fig = self.viz_module.create_energy_pattern_visualization(
                size=self.viz_size.value(),
                iterations=self.viz_iterations.value()
            )
            
        elif viz_index == 1:  # Cryptographic Visualization
            self.viz_canvas.fig = self.viz_module.create_cryptographic_visualization(
                text=self.viz_text.text()
            )
            
        elif viz_index == 2:  # LZ Convergence Animation
            # Create animation
            anim = self.viz_module.create_lz_convergence_animation(
                initial_values=[0.5, 1.0, 1.5, 2.0],
                max_iterations=self.viz_iterations.value()
            )
            
            # Store animation to prevent garbage collection
            self.current_animation = anim
            
            # Use the figure from the animation
            self.viz_canvas.fig = anim._fig
            
        elif viz_index == 3:  # Octave Mapping
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            
            # Create grid of values
            x = np.linspace(0, 2 * np.pi, self.viz_size.value())
            y = np.linspace(0, 2 * np.pi, self.viz_size.value())
            X, Y = np.meshgrid(x, y)
            
            # Calculate octave values
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # Calculate value based on position
                    val = np.sin(X[i, j]) * np.sin(Y[i, j]) * 10
                    
                    # Apply octave reduction
                    Z[i, j] = self.octave_module.octave_reduction(int(abs(val) * 100))
            
            # Create plot
            ax = fig.add_subplot(111)
            im = ax.imshow(Z, cmap='viridis', origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi])
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Octave')
            
            # Add labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Octave Mapping')
            
            self.viz_canvas.fig = fig
        
        # Update the canvas
        self.viz_canvas.draw()
        
        # Update status bar
        self.statusBar().showMessage('Visualization generated')
        
    def save_visualization(self):
        """Save the current visualization to a file."""
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Visualization',
            os.path.expanduser('~/com_visualization.png'),
            'Images (*.png *.jpg *.pdf)'
        )
        
        if file_path:
            # Save the figure
            self.viz_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            
            # Show confirmation
            self.statusBar().showMessage(f'Visualization saved to {file_path}')
            
    def add_about_tab(self):
        """Add the about tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Add title
        title = QLabel('Collatz Octave Model (COM) Framework')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        layout.addWidget(title)
        
        # Add description
        description = QLabel(
            'The COM Framework is based on the concept that reality is fundamentally '
            'energy-based with no vacuum or zero state. Space, time, mass, and forces '
            'are emergent properties of energy oscillations. The framework is built '
            'around the LZ constant (1.23498) and the HQS threshold (23.5% of LZ).\n\n'
            'This software provides tools for exploring the mathematical properties '
            'of the COM framework, analyzing octave patterns, and visualizing energy '
            'distributions.'
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add key concepts
        concepts_group = QGroupBox('Key Concepts')
        concepts_layout = QVBoxLayout(concepts_group)
        
        concepts = QLabel(
            '• LZ Constant (1.23498): A fixed point of the recursive wave function\n'
            '• Recursive Wave Function: f(x) = sin(x) + e^(-x)\n'
            '• HQS Threshold: 23.5% of LZ, represents phase transition boundary\n'
            '• Octave Reduction: Mapping numbers to values 1-9 based on digital root\n'
            '• Octave Scaling: Scaling by powers of LZ creates octave structure\n'
            '• Energy Patterns: Visualizations of energy distributions\n'
            '• Collatz-Octave Transform: Transformation based on Collatz sequence'
        )
        concepts.setWordWrap(True)
        concepts_layout.addWidget(concepts)
        layout.addWidget(concepts_group)
        
        # Add version info
        version_group = QGroupBox('Version Information')
        version_layout = QVBoxLayout(version_group)
        
        version = QLabel(
            'COM Framework Explorer v1.0\n'
            'Released: June © 2025 Martin Doina (dhelamay@protonmail.com)\n'
            'Python Version: 3.10\n'
            'Required Libraries: NumPy, SciPy, Matplotlib, PyQt5, Seaborn, pandas, Scikit-learn, mpmath'
        )
        version_layout.addWidget(version)
        layout.addWidget(version_group)
        
        # Add spacer
        layout.addStretch()
        
        # Add to tabs
        self.tabs.addTab(tab, 'About')


def main():
    """Main function to run the application."""
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    window = COMFrameworkApp()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
